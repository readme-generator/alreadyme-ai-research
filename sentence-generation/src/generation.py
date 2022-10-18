from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from collections import UserList
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from transformers import BatchEncoding, PreTrainedTokenizerBase

logger = logger.opt(colors=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class GenerationOptions:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int

    max_stream: int
    temperature: float
    pad_to_multiple_of: int
    max_length: int


class StreamSlots(UserList[str]):
    def __init__(self, num_slots: int):
        super().__init__([None for _ in range(num_slots)])

    def empty(self) -> bool:
        return all(slot is None for slot in self)

    def available(self) -> bool:
        return any(slot is None for slot in self)

    def assign(self, slot: str) -> int:
        if not self.available():
            raise ValueError("there are no available slots.")
        for i, item in enumerate(self):
            if item is None:
                self[i] = slot
                return i

    def free(self, index: int) -> str:
        slot = self[index]
        self[index] = None
        return slot


class GenerationContext:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
    ):
        attention_size = hidden_size // num_attention_heads
        past_shape = (batch_size, num_attention_heads, max_length - 1, attention_size)

        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
        self.past_key_values = [
            (torch.zeros(past_shape), torch.zeros(past_shape))
            for _ in range(num_hidden_layers)
        ]
        self.attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)

    @property
    def device(self) -> torch.device:
        return self.input_ids.device

    @property
    def dtype(self) -> torch.dtype:
        return self.past_key_values[0][0].dtype

    def to(self, target: Any) -> GenerationContext:
        self.past_key_values = [
            (k.to(target), v.to(target)) for k, v in self.past_key_values
        ]
        if not isinstance(target, torch.dtype):
            self.input_ids = self.input_ids.to(target)
            self.attention_mask = self.attention_mask.to(target)
        return self

    def type(self, dtype: torch.dtype) -> GenerationContext:
        return self.to(dtype)

    def cpu(self) -> GenerationContext:
        return self.to("cpu")

    def cuda(self) -> GenerationContext:
        return self.to("cuda")

    def float(self) -> GenerationContext:
        return self.to(torch.float32)

    def half(self) -> GenerationContext:
        return self.to(torch.float16)


@dataclass
class GenerationInitializer:
    model: torch.jit.ScriptModule
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8
    max_length: int = 2048

    def _update(
        self,
        context: GenerationContext,
        index: int,
        encodings: BatchEncoding,
        presents: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        input_length = encodings.attention_mask.sum().item()
        context.input_ids[index, 0] = encodings.input_ids[0, input_length - 1]
        context.attention_mask[index, :-input_length] = 0
        context.attention_mask[index, -input_length:] = 1

        for past_kv_pair, present_kv_pair in zip(context.past_key_values, presents):
            for past, present in zip(past_kv_pair, present_kv_pair):
                past[index, :, -input_length + 1 :, :].copy_(
                    present[0, :, : input_length - 1, :]
                )

    def _encode(self, input_prompt: str) -> BatchEncoding:
        if self.tokenizer.padding_side != "right":
            self.tokenizer.padding_side = "right"
        return self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def initialize(self, context: GenerationContext, index: int, input_prompt: str):
        encodings = self._encode(input_prompt).to(context.input_ids.device)
        presents = self.model(encodings.input_ids, [], encodings.attention_mask)[1]
        self._update(context, index, encodings, presents)


@dataclass
class GenerationLooper:
    model: torch.jit.ScriptModule
    temperature: float = 1.0

    def _update(
        self,
        context: GenerationContext,
        next_token_ids: torch.Tensor,
        presents: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        context.input_ids.copy_(next_token_ids)
        context.attention_mask.copy_(context.attention_mask.roll(-1, dims=1))
        context.attention_mask[:, -1] = context.attention_mask[:, -2]

        for past_kv_pair in context.past_key_values:
            for tensor in past_kv_pair:
                tensor.copy_(tensor.roll(-1, dims=2))
        for past_kv_pair, present_kv_pair in zip(context.past_key_values, presents):
            for past, present in zip(past_kv_pair, present_kv_pair):
                past[:, :, -1, :] = present[:, :, -1, :]

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        next_token_probs = (logits.float() / self.temperature).softmax(dim=2)
        next_token_ids = next_token_probs.squeeze(1).multinomial(1)
        return next_token_ids

    def generate(self, context: GenerationContext):
        logits, presents = self.model(
            context.input_ids, context.past_key_values, context.attention_mask
        )
        next_token_ids = self._sample(logits)
        self._update(context, next_token_ids, presents)


class GenerationSession:
    def __init__(
        self,
        model_path: str,
        tokenizer: PreTrainedTokenizerBase,
        options: GenerationOptions,
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.options = options

        self._session_closed = False
        self._stream_process = None
        self._stream_queues = {}

        self._request_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._termination_event = mp.Event()

    def initialize(self):
        self._session_closed = False
        asyncio.create_task(self._stream_response_fetcher_fn())

        self._stream_process = mp.Process(target=self._stream_generation_process_fn)
        self._stream_process.daemon = True
        self._stream_process.start()

    def close(self):
        if self._session_closed or self._stream_process is None:
            raise RuntimeError("session is already closed.")

        self._session_closed = True
        self._stream_queues.clear()

        self._termination_event.set()
        self._stream_process.join()

    def request(self, name: str, input_prompt: str):
        if name in self._stream_queues:
            raise RuntimeError(f"{name} is already requested.")
        self._stream_queues[name] = asyncio.Queue()
        self._request_queue.put_nowait((name, input_prompt))

    async def stream(self, name: str) -> AsyncGenerator[str]:
        while name in self._stream_queues:
            next_word = await self._stream_queues[name].get()
            if next_word is None:
                self._stream_queues.pop(name)
            else:
                yield next_word

    async def _stream_response_fetcher_fn(self):
        logger.info("asynchronous response fetching task has been started.")

        loop = asyncio.get_running_loop()
        while not self._session_closed:
            name, next_word = await loop.run_in_executor(None, self._response_queue.get)
            if next_word is not None:
                logger.debug(
                    f"response <y>{next_word.encode('unicode-escape').decode()}</y> is "
                    f"received for the request <r>{name}</r>."
                )
            if name in self._stream_queues:
                self._stream_queues[name].put_nowait(next_word)

        logger.info("asynchronous response fetching task is terminating.")

    @torch.inference_mode()
    def _stream_generation_process_fn(self):
        logger.info("generation subprocess has been started.")

        start_time = time.time()
        model = torch.jit.load(self.model_path).cuda().half()
        logger.info(
            f"<g>{self.model_path}</g> has been loaded. "
            f"estimated time: <g>{round(time.time() - start_time, 4)}</g>"
        )

        logger.info("creating generation context.")
        context = GenerationContext(
            batch_size=self.options.max_stream,
            max_length=self.options.max_length,
            hidden_size=self.options.hidden_size,
            num_hidden_layers=self.options.num_hidden_layers,
            num_attention_heads=self.options.num_attention_heads,
        )
        context.cuda().half()

        logger.info("creating initializer and looper.")
        initializer = GenerationInitializer(
            model,
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.options.pad_to_multiple_of,
            max_length=self.options.max_length,
        )
        looper = GenerationLooper(model, temperature=self.options.temperature)

        stream_slots = StreamSlots(self.options.max_stream)
        timestamps = [0 for _ in range(self.options.max_stream)]

        while not self._termination_event.is_set():
            if stream_slots.available() and not self._request_queue.empty():
                name, input_prompt = self._request_queue.get()
                index = stream_slots.assign(name)
                timestamps[index] = time.time()

                logger.info(
                    f"new generation request <r>{name}</r> is accepted because the "
                    f"stream slots are not full. <g>slot {index}</g> is assigned for "
                    f"<r>{name}</r> request."
                )

                start_time = time.time()
                initializer.initialize(context, index, input_prompt)
                logger.info(
                    f"generation context of the request <r>{name}</r> is successfully "
                    f"updated. estimated time: "
                    f"<g>{round(time.time() - start_time, 4)}</g>"
                )

            if stream_slots.empty():
                continue

            looper.generate(context)
            next_token_ids = context.input_ids.squeeze(1).tolist()
            complete_flags = context.attention_mask[:, 0].tolist()

            for i, (name, token_id, flag) in enumerate(
                zip(stream_slots, next_token_ids, complete_flags)
            ):
                if name is None:
                    continue
                if token_id != self.tokenizer.eos_token_id:
                    next_word = self.tokenizer.decode(token_id)
                    logger.debug(
                        f"<y>{next_word.encode('unicode-escape').decode()}</y> is "
                        f"generated for request {name}"
                    )
                    self._response_queue.put((name, next_word))
                if token_id == self.tokenizer.eos_token_id or flag == 1:
                    logger.info(
                        f"generation for the request <r>{name}</r> is completed "
                        f"because end-of-sentence is sampled or the sequence is "
                        f"reached to the maximum length. estimated time: "
                        f"<g>{time.time() - timestamps[i]}</g>"
                    )
                    self._response_queue.put((name, None))
                    stream_slots.free(i)

        logger.debug("terminating the generation subprocess.")


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        "sentence-generation/compiled-bloom-1b7-finetuned-readme-500k-steps"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-generation/compiled-bloom-1b7-finetuned-readme-500k-steps"
    )
    options = GenerationOptions(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        max_stream=4,
        temperature=0.87,
        pad_to_multiple_of=8,
        max_length=2048,
    )
    session = GenerationSession(
        "sentence-generation/compiled-bloom-1b7-finetuned-readme-500k-steps/model.pt",
        tokenizer,
        options,
    )

    with open("sentence-generation/legacy/prompt.txt") as fp:
        prompt = fp.read()

    async def write(name):
        text = ""
        async for word in session.stream(name):
            text += word
        with open(f"{name}.md", "w") as fp:
            fp.write(text)

    async def main():
        session.initialize()
        names = list(range(5))
        for name in names:
            session.request(name, prompt)
        tasks = [asyncio.create_task(write(name)) for name in names]
        await asyncio.wait(tasks)

    asyncio.run(main())
