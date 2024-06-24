from abc import ABCMeta, abstractmethod
import re

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def load_models(model_path: str, model_type: str):
    if model_type == "llava":
        return load_llava_models(model_path)
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")


def load_llava_models(model_path: str):
    return LlavaModel(model_path)


class VLModel(metaclass=ABCMeta):
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def encode_text(self, text: str):
        pass

    @abstractmethod
    def decode_text(self, output_ids: torch.tensor, skip_special_tokens: bool):
        pass

    @abstractmethod
    def encode_image(self, image: Image):
        pass

    @abstractmethod
    def generate(self, input_ids, images, image_sizes, do_sample, num_beams, max_new_tokens, use_cache):
        pass


class LlavaModel(VLModel):
    def __init__(self, model_path: str):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, model_base=None, model_name=get_model_name_from_path(model_path)
        )
        self.model_name = get_model_name_from_path(model_path)
        self.conv_mode = self._conv_mode()

    def to(self, device):
        self.model.to(device)
        self.device = device

    def eval(self):
        self.model.eval()

    def encode_text(self, text: str):
        input_ids = (
            tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        return input_ids

    def decode_text(self, output_ids: torch.tensor, skip_special_tokens: bool):
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)

    def encode_image(self, image: Image):
        image_tensor = process_images([image], self.image_processor, self.model.config).to(
            self.device, dtype=torch.float16
        )
        image_size = [image_tensor.size]
        return image_tensor, image_size

    def build_prompt(self, source: str, target: str):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in source:
            if self.model.config.mm_use_im_start_end:
                source = re.sub(IMAGE_PLACEHOLDER, image_token_se, source)
            else:
                source = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, source)
        else:
            if self.model.config.mm_use_im_start_end:
                source = image_token_se + "\n" + source
            else:
                source = DEFAULT_IMAGE_TOKEN + "\n" + source
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], source)
        conv.append_message(conv.roles[1], target)
        prompt = conv.get_prompt()
        return prompt

    def _conv_mode(self):
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        print("conv_mode:", conv_mode)
        return conv_mode

    def __call__(self, input_ids, images, image_sizes, labels):
        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                labels=labels,
            )
        return outputs

    def generate(self, input_ids, images, image_sizes, **kwargs):
        output_ids = self.model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=kwargs["do_sample"],
            max_new_tokens=kwargs["max_new_tokens"],
            use_cache=kwargs["use_cache"],
            temperature=kwargs["temperature"],
        )
        return output_ids


# class InstructBlipModel(VLModel):
#     # Implement here
#     def __init__(self, model_path: str):
#         model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
#         processor = InstructBlipProcessor.from_pretrained(model_path)
#         self.model = model
#         self.processor = processor

#     def to(self, device):
#         self.model.to(device)
#         self.device = device

#     def eval(self):
#         self.model.eval()

#     def encode_text(self, text: str):
#         input_ids = (
#             tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .to(self.device)
#         )
#         return input_ids

#     def decode_text(self, output_ids: torch.tensor, skip_special_tokens: bool):
#         return self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)

#     def encode_image(self, image: Image):
#         image_tensor = process_images([image], self.image_processor, self.model.config).to(
#             self.device, dtype=torch.float16
#         )
#         image_size = [image_tensor.size]
#         return image_tensor, image_size

#     def build_prompt(self, source: str, target: str):
#         image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#         if IMAGE_PLACEHOLDER in source:
#             if self.model.config.mm_use_im_start_end:
#                 source = re.sub(IMAGE_PLACEHOLDER, image_token_se, source)
#             else:
#                 source = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, source)
#         else:
#             if self.model.config.mm_use_im_start_end:
#                 source = image_token_se + "\n" + source
#             else:
#                 source = DEFAULT_IMAGE_TOKEN + "\n" + source
#         conv = conv_templates[self.conv_mode].copy()
#         conv.append_message(conv.roles[0], source)
#         conv.append_message(conv.roles[1], target)
#         prompt = conv.get_prompt()
#         return prompt

#     def _conv_mode(self):
#         if "llama-2" in self.model_name.lower():
#             conv_mode = "llava_llama_2"
#         elif "mistral" in self.model_name.lower():
#             conv_mode = "mistral_instruct"
#         elif "v1.6-34b" in self.model_name.lower():
#             conv_mode = "chatml_direct"
#         elif "v1" in self.model_name.lower():
#             conv_mode = "llava_v1"
#         elif "mpt" in self.model_name.lower():
#             conv_mode = "mpt"
#         else:
#             conv_mode = "llava_v0"

#         print("conv_mode:", conv_mode)
#         return conv_mode

#     def __call__(self, input_ids, images, image_sizes, labels):
#         with torch.inference_mode():
#             outputs = self.model(
#                 input_ids,
#                 images=images,
#                 image_sizes=image_sizes,
#                 labels=labels,
#             )
#         return outputs

#     def generate(self, input_ids, images, image_sizes, **kwargs):
#         output_ids = self.model.generate(
#             input_ids,
#             images=images,
#             image_sizes=image_sizes,
#             do_sample=kwargs["do_sample"],
#             max_new_tokens=kwargs["max_new_tokens"],
#             use_cache=kwargs["use_cache"],
#             temperature=kwargs["temperature"],
#         )
#         return output_ids