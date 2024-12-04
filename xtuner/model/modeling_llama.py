from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List

# 定义子类，继承自 LlamaForCausalLM
class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 你可以在这里添加额外的初始化代码

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 调用父类的 forward 方法，但不传递 labels 参数，避免父类计算 loss
        # outputs = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     cache_position=cache_position,
        #     num_logits_to_keep=num_logits_to_keep,
        #     **loss_kwargs,
        # )
        #
        # # 获取 logits
        # logits = outputs.logits
        #
        # # 重新计算 loss
        # loss = None
        # if labels is not None:
        #     # 自定义 loss 计算
        #     loss = self.custom_loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
        #
        # # 返回修改后的输出
        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output
        #
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            weight_tensors = self.find_num(labels)
            loss = self.custom_loss_function(logits=logits, labels=labels, weight_tensors=weight_tensors, vocab_size=self.config.vocab_size, **loss_kwargs)
            # loss = self.custom_loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), hidden_states

    def fixed_cross_entropy(self, source, target, weight_tensors, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        reduction = "sum" if num_items_in_batch is not None else "mean"
        if weight_tensors is not None:
            reduction = "none"

        weight_tensors = weight_tensors / torch.sum(target!=-100)
        loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction) * weight_tensors
        if reduction == "sum":
            loss = loss / num_items_in_batch
        if reduction == "none":
            loss = loss.sum()
        return loss

    def find_num(self, tensors):

        # 初始化 weight_tensor
        weight_tensors = torch.ones_like(tensors)

        # 查找第一个出现 910, 338, 263, 1855 或 910, 338, 263, 25713 模式的位置

        for j in range(tensors.shape[0]):
            i = 0
            tensor = tensors[j]
            weight_tensor = weight_tensors[j]
            while i < len(tensor) - 3:
                if tensor[i] == -100:
                    i += 1
                    continue
                if tensor[i] == 910 and tensor[i + 1] == 338 and tensor[i + 2] == 263:
                    if tensor[i + 3] == 1855 or tensor[i + 3] == 25713:
                        weight_tensor[i + 3] = 3
                        break
                i += 1
        return weight_tensors
            # print("Weight Tensor:")
            # print(weight_tensor)

    def custom_loss_function(
            self, logits, labels, weight_tensors, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weight_tensors = weight_tensors[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_weight_tensors = shift_weight_tensors.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        shift_weight_tensors = shift_weight_tensors.to(shift_logits.device)
        loss = self.fixed_cross_entropy(shift_logits, shift_labels, shift_weight_tensors, num_items_in_batch, ignore_index, **kwargs)
        return loss

# 示例使用
if __name__ == "__main__":
    # 加载预训练的 Llama 模型和 tokenizer
    model_name = '/data/kesun/vicuna-7b-v1.5'  # 替换为你的预训练模型名称
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = CustomLlamaForCausalLM.from_pretrained(model_name)

    # 准备输入数据
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])  # 示例标签

    # 前向传播
    outputs = model(**inputs, labels=labels)

    # 打印输出
    print(outputs)
