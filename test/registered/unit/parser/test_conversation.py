# 文件名: test_conversation.py - 对话
"""Unit tests for srt/parser/conversation.py"""

import json
import os
import tempfile
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentAudioPart,
    ChatCompletionMessageContentAudioURL,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentVideoURL,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    _get_full_multimodal_text_prompt,
    chat_template_exists,
    chat_templates,
    generate_chat_conv,
    generate_embedding_convs,
    get_conv_template_by_model_path,
    get_model_type,
    register_conv_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


# TestConversationGetPrompt类
class TestConversationGetPrompt(CustomTestCase):

    # TestConversationGetPrompt类的测试addcolonsingle
    def test_add_colon_single(self):
        """Test prompt generation with ADD_COLON_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="System msg",
            roles=("User", "Assistant"),
            messages=[["User", "Hello"], ["Assistant", "Hi"], ["User", None]],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("System msg\n", prompt)  # 断言包含
        self.assertIn("User: Hello\n", prompt)  # 断言包含
        self.assertIn("Assistant: Hi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("User:"))  # 断言为真

    # TestConversationGetPrompt类的测试addcolontwo
    def test_add_colon_two(self):
        """Test prompt generation with ADD_COLON_TWO style (alternating separators)."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", "A"], ["User", None]],
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep="<s1>",
            sep2="<s2>",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Q<s1>", prompt)  # 断言包含
        self.assertIn("Assistant: A<s2>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("User:"))  # 断言为真

    # TestConversationGetPrompt类的测试chatml
    def test_chatml(self):
        """Test prompt generation with CHATML style."""
        conv = Conversation(
            name="test",
            system_message="<|im_start|>system\nYou are helpful",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            messages=[
                ["<|im_start|>user", "Hello"],
                ["<|im_start|>assistant", None],
            ],
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        )
        prompt = conv.get_prompt()
        self.assertIn("You are helpful<|im_end|>", prompt)  # 断言包含
        self.assertIn("<|im_start|>user\nHello<|im_end|>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))  # 断言为真

    # TestConversationGetPrompt类的测试llama3
    def test_llama3(self):
        """Test prompt generation with LLAMA3 style."""
        conv = Conversation(
            name="test",
            system_message="<|start_header_id|>system<|end_header_id|>\n\nBe helpful<|eot_id|>",
            roles=("user", "assistant"),
            messages=[["user", "Hi"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA3,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful<|eot_id|>", prompt)  # 断言包含
        self.assertIn(  # 断言包含
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>", prompt
        )
        self.assertTrue(  # 断言为真
            prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        )

    # TestConversationGetPrompt类的测试nocolonsingle
    def test_no_colon_single(self):
        """Test prompt generation with NO_COLON_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[USER]", "[ASST]"),
            messages=[["[USER]", "Hello"], ["[ASST]", None]],
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[USER]Hello\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("[ASST]"))  # 断言为真

    # TestConversationGetPrompt类的测试nonemessageinprompt
    def test_none_message_in_prompt(self):
        """Test that None message produces role-only output (no content)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.endswith("Assistant:"))  # 断言为真

    # TestConversationGetPrompt类的测试emptysystemmessage
    def test_empty_system_message(self):
        """Test that empty system message produces empty prefix for LLAMA3."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Hello"], ["Assistant", None]],
            sep_style=SeparatorStyle.LLAMA3,
        )
        prompt = conv.get_prompt()
        self.assertNotIn("system", prompt.lower())  # 断言不包含

    # TestConversationGetPrompt类的测试addcolonspacesingle
    def test_add_colon_space_single(self):
        """Test prompt generation with ADD_COLON_SPACE_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Bot"),
            messages=[["User", "Hi"], ["Bot", None]],
            sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Hi\n", prompt)  # 断言包含
        # None message should end with ": " (space after colon)
        self.assertTrue(prompt.endswith("Bot: "))  # 断言为真

    # TestConversationGetPrompt类的测试addnewlinesingle
    def test_add_new_line_single(self):
        """Test prompt generation with ADD_NEW_LINE_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Bot"),
            messages=[["User", "Hi"], ["Bot", None]],
            sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("User\nHi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("Bot\n"))  # 断言为真

    # TestConversationGetPrompt类的测试nocolontwo
    def test_no_colon_two(self):
        """Test prompt generation with NO_COLON_TWO style (alternating separators)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[U]", "[A]"),
            messages=[["[U]", "Q"], ["[A]", "A"], ["[U]", None]],
            sep_style=SeparatorStyle.NO_COLON_TWO,
            sep="<s1>",
            sep2="<s2>",
        )
        prompt = conv.get_prompt()
        self.assertIn("[U]Q<s1>", prompt)  # 断言包含
        self.assertIn("[A]A<s2>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("[U]"))  # 断言为真

    # TestConversationGetPrompt类的测试llama2withsystem
    def test_llama2_with_system(self):
        """Test LLAMA2 with system message."""
        conv = Conversation(
            name="test",
            system_message="<<SYS>>\nBe helpful\n<</SYS>>\n\n",
            system_template="[INST] {system_message}",
            roles=("[INST]", "[/INST]"),
            messages=[["[INST]", "Hi"], ["[/INST]", None]],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)  # 断言包含
        self.assertIn("Hi ", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试llama2withoutsystem
    def test_llama2_without_system(self):
        """Test LLAMA2 without system message falls back to '[INST] ' prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[INST]", "[/INST]"),
            messages=[["[INST]", "Hi"], ["[/INST]", None]],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("[INST] Hi"))  # 断言为真

    # TestConversationGetPrompt类的测试llama2multiturn
    def test_llama2_multi_turn(self):
        """Test LLAMA2 with multi-turn (i>0 uses tag+sep pattern)."""
        conv = Conversation(
            name="test",
            system_message="<<SYS>>\nSys\n<</SYS>>\n\n",
            system_template="[INST] {system_message}",
            roles=("[INST]", "[/INST]"),
            messages=[
                ["[INST]", "Q1"],
                ["[/INST]", "A1"],
                ["[INST]", "Q2"],
                ["[/INST]", None],
            ],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        # i=0: message + " " (no tag prefix)
        self.assertIn("Q1 ", prompt)  # 断言包含
        # i=1: tag + " " + message + sep2
        self.assertIn("[/INST] A1 </s><s>", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试llama4
    def test_llama4(self):
        """Test prompt generation with LLAMA4 style."""
        conv = Conversation(
            name="test",
            system_message="Be helpful",
            system_template="{system_message}",
            roles=("user", "assistant"),
            messages=[["user", "Hello"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA4,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)  # 断言包含
        self.assertIn("<|header_start|>user<|header_end|>", prompt)  # 断言包含
        self.assertIn("Hello<|eot|>", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试llama4emptysystem
    def test_llama4_empty_system(self):
        """Test LLAMA4 with empty system message omits system prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("user", "assistant"),
            messages=[["user", "Hello"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA4,
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("<|header_start|>user"))  # 断言为真

    # TestConversationGetPrompt类的测试chatglm3
    def test_chatglm3(self):
        """Test prompt generation with CHATGLM3 style."""
        conv = Conversation(
            name="test",
            system_message="<|system|>\nBe helpful",
            roles=("<|user|>", "<|assistant|>"),
            messages=[["<|user|>", "Hi"], ["<|assistant|>", None]],
            sep_style=SeparatorStyle.CHATGLM3,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)  # 断言包含
        self.assertIn("<|user|>\nHi", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("<|assistant|>"))  # 断言为真

    # TestConversationGetPrompt类的测试deepseekchat
    def test_deepseek_chat(self):
        """Test prompt generation with DEEPSEEK_CHAT style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", "A"], ["User", None]],
            sep_style=SeparatorStyle.DEEPSEEK_CHAT,
            sep="\n\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Q\n\n", prompt)  # 断言包含
        self.assertIn("Assistant: A<end>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("User:"))  # 断言为真

    # TestConversationGetPrompt类的测试robin
    def test_robin(self):
        """Test prompt generation with ROBIN style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("###Human", "###Assistant"),
            messages=[["###Human", "Hi"], ["###Assistant", None]],
            sep_style=SeparatorStyle.ROBIN,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("###Human:\nHi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("###Assistant:\n"))  # 断言为真

    # TestConversationGetPrompt类的测试falconchat
    def test_falcon_chat(self):
        """Test prompt generation with FALCON_CHAT style."""
        conv = Conversation(
            name="test",
            system_message="System prompt.",
            roles=("User", "Falcon"),
            messages=[["User", "Hi"], ["Falcon", None]],
            sep_style=SeparatorStyle.FALCON_CHAT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("System prompt.\n", prompt)  # 断言包含
        self.assertIn("User: Hi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("Falcon:"))  # 断言为真

    # TestConversationGetPrompt类的测试metamath
    def test_metamath(self):
        """Test prompt generation with METAMATH style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Query", "Response"),
            messages=[["Query", "2+2?"], ["Response", None]],
            sep_style=SeparatorStyle.METAMATH,
            sep="\n",
            sep2="Let's think step by step.\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("Query:\n2+2?\n", prompt)  # 断言包含
        self.assertIn("Response: Let's think step by step.\n", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试mpt
    def test_mpt(self):
        """Test prompt generation with MPT style."""
        conv = Conversation(
            name="test",
            system_message="<|system|>",
            roles=("<|user|>", "<|assistant|>"),
            messages=[["<|user|>", "Hi"], ["<|assistant|>", None]],
            sep_style=SeparatorStyle.MPT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("<|user|>Hi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("<|assistant|>"))  # 断言为真

    # TestConversationGetPrompt类的测试chatintern
    def test_chatintern(self):
        """Test prompt generation with CHATINTERN style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("HUMAN", "BOT"),
            messages=[["HUMAN", "Hi"], ["BOT", "Hello"], ["HUMAN", None]],
            sep_style=SeparatorStyle.CHATINTERN,
            sep="\n",
            sep2="</s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("<s>HUMAN:Hi\n", prompt)  # 断言包含
        self.assertIn("BOT:Hello</s>", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试dolly
    def test_dolly(self):
        """Test prompt generation with DOLLY style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Instruction", "Response"),
            messages=[["Instruction", "Q"], ["Response", "A"], ["Instruction", None]],
            sep_style=SeparatorStyle.DOLLY,
            sep="\n\n",
            sep2="</s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Instruction:\nQ\n\n", prompt)  # 断言包含
        self.assertIn("Response:\nA</s>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("Instruction:\n"))  # 断言为真

    # TestConversationGetPrompt类的测试phoenix
    def test_phoenix(self):
        """Test prompt generation with PHOENIX style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Human", "Phoenix"),
            messages=[["Human", "Hi"], ["Phoenix", None]],
            sep_style=SeparatorStyle.PHOENIX,
        )
        prompt = conv.get_prompt()
        self.assertIn("Human: <s>Hi</s>", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("Phoenix: <s>"))  # 断言为真

    # TestConversationGetPrompt类的测试deepseekvl2
    def test_deepseek_vl2(self):
        """Test prompt generation with DeepSeekVL2 style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.DeepSeekVL2,
            sep="\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Sys\n", prompt)  # 断言包含
        self.assertIn("User: Q\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("Assistant:"))  # 断言为真

    # TestConversationGetPrompt类的测试deepseekvl2emptysystem
    def test_deepseek_vl2_empty_system(self):
        """Test DeepSeekVL2 with empty system message omits system prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.DeepSeekVL2,
            sep="\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("User: Q"))  # 断言为真

    # TestConversationGetPrompt类的测试gemma3
    def test_gemma3(self):
        """Test prompt generation with GEMMA3 style (first message special)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("<start>", "<model>"),
            messages=[["<start>", "Hello"], ["<model>", "Hi"], ["<start>", None]],
            sep_style=SeparatorStyle.GEMMA3,
            sep="<end>",
        )
        prompt = conv.get_prompt()
        # First message: no role prefix, just message + sep
        self.assertTrue(prompt.startswith("Hello<end>"))  # 断言为真
        # Subsequent: role + message + sep
        self.assertIn("<model>Hi<end>", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试rwkv
    def test_rwkv(self):
        """Test prompt generation with RWKV style (newline replacement)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Bob", "Alice"),
            messages=[["Bob", "Hello\n\nWorld"], ["Alice", None]],
            sep_style=SeparatorStyle.RWKV,
        )
        prompt = conv.get_prompt()
        # RWKV replaces \n\n with \n in message
        self.assertIn("Bob: Hello\nWorld\n\n", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试qwen2vlembed
    def test_qwen2_vl_embed(self):
        """Test prompt generation with QWEN2_VL_EMBED style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("user", "assistant"),
            messages=[["user", "Hi"], ["assistant", None]],
            sep_style=SeparatorStyle.QWEN2_VL_EMBED,
            sep="\n",
            stop_str="<|endoftext|>",
        )
        prompt = conv.get_prompt()
        self.assertIn("user\nHi\n", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("<|endoftext|>"))  # 断言为真

    # TestConversationGetPrompt类的测试chatglm
    def test_chatglm(self):
        """Test prompt generation with CHATGLM style (round numbering)."""
        conv = Conversation(
            name="chatglm",
            system_message="",
            roles=("问", "答"),
            messages=[["问", "Hello"], ["答", "Hi"], ["问", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[Round 0]\n", prompt)  # 断言包含
        self.assertIn("问：Hello\n", prompt)
        self.assertIn("答：Hi\n", prompt)
        self.assertTrue(prompt.endswith("问："))

    # TestConversationGetPrompt类的测试chatglm2roundoffset
    def test_chatglm2_round_offset(self):
        """Test CHATGLM style with chatglm2 name (round starts at 1 instead of 0)."""
        conv = Conversation(
            name="chatglm2",
            system_message="",
            roles=("问", "答"),
            messages=[["问", "Hello"], ["答", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[Round 1]\n", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试chatglmwithsystem
    def test_chatglm_with_system(self):
        """Test CHATGLM with non-empty system message."""
        conv = Conversation(
            name="chatglm",
            system_message="You are helpful",
            roles=("问", "答"),
            messages=[["问", "Hi"], ["答", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("You are helpful\n"))  # 断言为真

    # TestConversationGetPrompt类的测试qwen2audio
    def test_qwen2_audio(self):
        """Test QWEN2_AUDIO style with audio token counter replacement."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("user", "assistant"),
            messages=[
                ["user", "Listen: <audio>{idx}</audio> and <audio>{idx}</audio>"],
                ["assistant", None],
            ],
            sep_style=SeparatorStyle.QWEN2_AUDIO,
            sep="\n",
            audio_token="<audio>{idx}</audio>",
        )
        prompt = conv.get_prompt()
        # Audio tokens should be replaced with counter: idx=1, idx=2
        self.assertIn("<audio>1</audio>", prompt)  # 断言包含
        self.assertIn("<audio>2</audio>", prompt)  # 断言包含
        self.assertNotIn("{idx}", prompt)  # 断言不包含

    # TestConversationGetPrompt类的测试paddleocr
    def test_paddle_ocr(self):
        """Test prompt generation with PADDLE_OCR style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("USER", "ASSISTANT"),
            messages=[["USER", "Describe image"], ["ASSISTANT", None]],
            sep_style=SeparatorStyle.PADDLE_OCR,
            sep="<eos>",
        )
        prompt = conv.get_prompt()
        self.assertIn("USER: Describe image", prompt)  # 断言包含
        self.assertTrue(prompt.endswith("ASSISTANT: "))  # 断言为真

    # TestConversationGetPrompt类的测试paddleocrwithimagetoken
    def test_paddle_ocr_with_image_token(self):
        """Test PADDLE_OCR strips newline after image token for USER role."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("USER", "ASSISTANT"),
            messages=[
                ["USER", "<image>\nDescribe this"],
                ["ASSISTANT", "It shows a cat"],
            ],
            sep_style=SeparatorStyle.PADDLE_OCR,
            sep="<eos>",
            image_token="<image>",
        )
        prompt = conv.get_prompt()
        # image_token + "\n" should be replaced with just image_token
        self.assertIn("USER: <image>Describe this\n", prompt)  # 断言包含
        self.assertIn("ASSISTANT: It shows a cat<eos>", prompt)  # 断言包含

    # TestConversationGetPrompt类的测试mptwithtuplemessage
    def test_mpt_with_tuple_message(self):
        """Test MPT style extracts first element from tuple messages."""
        conv = Conversation(
            name="test",
            system_message="<|system|>",
            roles=("<|user|>", "<|assistant|>"),
            messages=[
                ["<|user|>", ("Hello", "extra1", "extra2")],
                ["<|assistant|>", None],
            ],
            sep_style=SeparatorStyle.MPT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("<|user|>Hello\n", prompt)  # 断言包含
        self.assertNotIn("extra1", prompt)  # 断言不包含

    # TestConversationGetPrompt类的测试invalidsepstyleraises
    def test_invalid_sep_style_raises(self):
        """Test that an invalid SeparatorStyle raises ValueError."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("A", "B"),
            messages=[["A", "Hi"]],
            sep_style=999,
            sep="\n",
        )
        with self.assertRaises(ValueError):  # 断言抛出异常
            conv.get_prompt()


# TestConversationMethods类
class TestConversationMethods(CustomTestCase):

    # TestConversationMethods类的内部方法_make_conv
    def _make_conv(self):
        return Conversation(
            name="test",
            roles=("User", "Assistant"),
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )

    # TestConversationMethods类的测试appendmessage
    def test_append_message(self):
        """Test appending messages to conversation."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", "Hi")
        self.assertEqual(len(conv.messages), 2)  # 断言相等
        self.assertEqual(conv.messages[0], ["User", "Hello"])  # 断言相等

    # TestConversationMethods类的测试setsystemmessage
    def test_set_system_message(self):
        """Test setting the system message."""
        conv = self._make_conv()
        conv.set_system_message("Be helpful")
        self.assertEqual(conv.system_message, "Be helpful")  # 断言相等

    # TestConversationMethods类的测试updatelastmessage
    def test_update_last_message(self):
        """Test updating the last message in-place."""
        conv = self._make_conv()
        conv.append_message("User", "Q")
        conv.append_message("Assistant", None)
        conv.update_last_message("Answer")
        self.assertEqual(conv.messages[-1][1], "Answer")  # 断言相等

    # TestConversationMethods类的测试toopenaiapimessageswithsystem
    def test_to_openai_api_messages_with_system(self):
        """Test conversion to OpenAI format with system message."""
        conv = self._make_conv()
        conv.system_message = "Be helpful"
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", "Hi")
        result = conv.to_openai_api_messages()
        self.assertEqual(result[0], {"role": "system", "content": "Be helpful"})  # 断言相等
        self.assertEqual(result[1], {"role": "user", "content": "Hello"})  # 断言相等
        self.assertEqual(result[2], {"role": "assistant", "content": "Hi"})  # 断言相等

    # TestConversationMethods类的测试toopenaiapimessageswithoutsystem
    def test_to_openai_api_messages_without_system(self):
        """Test conversion to OpenAI format without system message."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        result = conv.to_openai_api_messages()
        self.assertEqual(len(result), 1)  # 断言相等
        self.assertEqual(result[0]["role"], "user")  # 断言相等

    # TestConversationMethods类的测试toopenaiapimessagesskipsnoneassistant
    def test_to_openai_api_messages_skips_none_assistant(self):
        """Test that None assistant message is omitted from OpenAI format."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", None)
        result = conv.to_openai_api_messages()
        self.assertEqual(len(result), 1)  # only user message  # 断言相等

    # TestConversationMethods类的测试togradiochatbot
    def test_to_gradio_chatbot(self):
        """Test conversion to Gradio chatbot format (user/assistant pairs)."""
        conv = self._make_conv()
        conv.append_message("User", "Q1")
        conv.append_message("Assistant", "A1")
        conv.append_message("User", "Q2")
        conv.append_message("Assistant", "A2")
        result = conv.to_gradio_chatbot()
        self.assertEqual(len(result), 2)  # 断言相等
        self.assertEqual(result[0], ["Q1", "A1"])  # 断言相等
        self.assertEqual(result[1], ["Q2", "A2"])  # 断言相等

    # TestConversationMethods类的测试togradiochatbotpendingresponse
    def test_to_gradio_chatbot_pending_response(self):
        """Test Gradio format with pending assistant response (None)."""
        conv = self._make_conv()
        conv.append_message("User", "Q1")
        conv.append_message("Assistant", None)
        result = conv.to_gradio_chatbot()
        self.assertEqual(result, [["Q1", None]])  # 断言相等

    # TestConversationMethods类的测试appendimage
    def test_append_image(self):
        """Test appending image data to conversation."""
        conv = self._make_conv()
        conv.image_data = []
        conv.append_image("http://example.com/img.jpg", "auto")
        self.assertEqual(len(conv.image_data), 1)  # 断言相等
        self.assertEqual(conv.image_data[0].url, "http://example.com/img.jpg")  # 断言相等
        self.assertEqual(conv.image_data[0].detail, "auto")  # 断言相等

    # TestConversationMethods类的测试appendvideo
    def test_append_video(self):
        """Test appending video data to conversation."""
        conv = self._make_conv()
        conv.video_data = []
        conv.append_video("http://example.com/vid.mp4")
        self.assertEqual(len(conv.video_data), 1)  # 断言相等
        self.assertEqual(conv.video_data[0], "http://example.com/vid.mp4")  # 断言相等

    # TestConversationMethods类的测试appendaudio
    def test_append_audio(self):
        """Test appending audio data to conversation."""
        conv = self._make_conv()
        conv.audio_data = []
        conv.append_audio("http://example.com/audio.wav")
        self.assertEqual(len(conv.audio_data), 1)  # 断言相等
        self.assertEqual(conv.audio_data[0], "http://example.com/audio.wav")  # 断言相等

    # TestConversationMethods类的测试copyisindependent
    def test_copy_is_independent(self):
        """Test that copy() creates an independent conversation."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        copied = conv.copy()
        copied.append_message("Assistant", "Hi")
        self.assertEqual(len(conv.messages), 1)  # 断言相等
        self.assertEqual(len(copied.messages), 2)  # 断言相等

    # TestConversationMethods类的测试dictserialization
    def test_dict_serialization(self):
        """Test dict() returns expected keys."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        d = conv.dict()
        self.assertEqual(d["template_name"], "test")  # 断言相等
        self.assertIn("messages", d)  # 断言包含
        self.assertIn("roles", d)  # 断言包含


# TestTemplateRegistry类
class TestTemplateRegistry(CustomTestCase):

    # TestTemplateRegistry类的测试builtintemplatesexist
    def test_builtin_templates_exist(self):
        """Test that common built-in templates are registered."""
        self.assertTrue(chat_template_exists("chatml"))  # 断言为真
        self.assertTrue(chat_template_exists("llama-2"))  # 断言为真

    # TestTemplateRegistry类的测试unregisteredtemplatenotfound
    def test_unregistered_template_not_found(self):
        """Test that non-existent template returns False."""
        self.assertFalse(chat_template_exists("_nonexistent_template_xyz"))  # 断言为假

    # TestTemplateRegistry类的测试registerandlookup
    def test_register_and_lookup(self):
        """Test registering and looking up a custom template."""
        t = Conversation(
            name="_test_conv_template",
            roles=("A", "B"),
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        register_conv_template(t)
        self.assertTrue(chat_template_exists("_test_conv_template"))  # 断言为真
        # Cleanup
        del chat_templates["_test_conv_template"]

    # TestTemplateRegistry类的测试registerduplicateraises
    def test_register_duplicate_raises(self):
        """Test that registering a duplicate name without override raises."""
        with self.assertRaises(AssertionError):  # 断言抛出异常
            register_conv_template(
                Conversation(
                    name="chatml",
                    roles=("A", "B"),
                    messages=[],
                    sep_style=SeparatorStyle.CHATML,
                    sep="",
                )
            )

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathreturnsnoneforunknown
    def test_get_conv_template_by_model_path_returns_none_for_unknown(self):
        """Test that unknown model path returns None."""
        result = get_conv_template_by_model_path("totally-unknown-model-xyz")
        self.assertIsNone(result)  # 断言为None

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathvicuna
    def test_get_conv_template_by_model_path_vicuna(self):
        """Test that vicuna model path is matched correctly."""
        result = get_conv_template_by_model_path("lmsys/vicuna-7b-v1.5")
        self.assertEqual(result, "vicuna_v1.1")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathinternvl
    def test_get_conv_template_by_model_path_internvl(self):
        """Test that internvl model path is matched correctly."""
        result = get_conv_template_by_model_path("OpenGVLab/InternVL2-8B")
        self.assertEqual(result, "internvl-2-5")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathdeepseekvl2
    def test_get_conv_template_by_model_path_deepseek_vl2(self):
        """Test that deepseek-vl2 model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/deepseek-vl2")
        self.assertEqual(result, "deepseek-vl2")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathwhisper
    def test_get_conv_template_by_model_path_whisper(self):
        """Test that whisper model path is matched correctly."""
        result = get_conv_template_by_model_path("openai/whisper-large-v3")
        self.assertEqual(result, "whisper")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathjanus
    def test_get_conv_template_by_model_path_janus(self):
        """Test that janus model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/Janus-Pro-7B")
        self.assertEqual(result, "janus-pro")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathphi4mm
    def test_get_conv_template_by_model_path_phi4_mm(self):
        """Test that phi-4-multimodal model path is matched correctly."""
        result = get_conv_template_by_model_path("microsoft/phi-4-multimodal")
        self.assertEqual(result, "phi-4-mm")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathllavanext
    def test_get_conv_template_by_model_path_llava_next(self):
        """Test that llava-next-video-34b model path returns chatml-llava."""
        result = get_conv_template_by_model_path("llava-hf/llava-next-video-34b")
        self.assertEqual(result, "chatml-llava")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathpaddleocr
    def test_get_conv_template_by_model_path_paddle_ocr(self):
        """Test that paddleocr model path is matched correctly."""
        result = get_conv_template_by_model_path("PaddleOCR/PaddleOCR-2.9")
        self.assertEqual(result, "paddle-ocr")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathdeepseekocr
    def test_get_conv_template_by_model_path_deepseek_ocr(self):
        """Test that deepseek-ocr model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/deepseek-ocr-base")
        self.assertEqual(result, "deepseek-ocr")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathpoints
    def test_get_conv_template_by_model_path_points(self):
        """Test that points model path is matched correctly."""
        result = get_conv_template_by_model_path("WePOINTS/points-v1.5")
        self.assertEqual(result, "points-v15-chat")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathminicpmv
    def test_get_conv_template_by_model_path_minicpm_v(self):
        """Test that minicpm-v model path returns minicpmv."""
        result = get_conv_template_by_model_path("openbmb/MiniCPM-V-2_6")
        self.assertEqual(result, "minicpmv")  # 断言相等

    # TestTemplateRegistry类的测试getconvtemplatebymodelpathminicpmo
    def test_get_conv_template_by_model_path_minicpm_o(self):
        """Test that minicpm-o model path returns minicpmo."""
        result = get_conv_template_by_model_path("openbmb/MiniCPM-o-2_6")
        self.assertEqual(result, "minicpmo")  # 断言相等


# TestGenerateEmbeddingConvs类
class TestGenerateEmbeddingConvs(CustomTestCase):

    # TestGenerateEmbeddingConvs类的测试textonly
    def test_text_only(self):
        """Test generating embedding conversations with text only."""
        convs = generate_embedding_convs(
            texts=["Hello world"],
            images=[None],
            videos=[None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)  # 断言相等
        self.assertEqual(len(convs[0].messages), 2)  # 断言相等
        self.assertIn("Hello world", convs[0].messages[0][1])  # 断言包含
        self.assertIsNone(convs[0].messages[1][1])  # assistant placeholder  # 断言为None

    # TestGenerateEmbeddingConvs类的测试withimage
    def test_with_image(self):
        """Test generating embedding conversations with image."""
        convs = generate_embedding_convs(
            texts=["Describe"],
            images=["http://example.com/img.jpg"],
            videos=[None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)  # 断言相等
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)  # 断言包含
        self.assertIn("Describe", msg)  # 断言包含

    # TestGenerateEmbeddingConvs类的测试withvideo
    def test_with_video(self):
        """Test generating embedding conversations with video."""
        convs = generate_embedding_convs(
            texts=["Describe"],
            images=[None],
            videos=["http://example.com/vid.mp4"],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)  # 断言相等
        msg = convs[0].messages[0][1]
        self.assertIn("<video>", msg)  # 断言包含
        self.assertIn("Describe", msg)  # 断言包含

    # TestGenerateEmbeddingConvs类的测试withimageandvideo
    def test_with_image_and_video(self):
        """Test embedding conv with both image and video."""
        convs = generate_embedding_convs(
            texts=["Desc"],
            images=["http://example.com/img.jpg"],
            videos=["http://example.com/vid.mp4"],
            template_name="chatml",
        )
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)  # 断言包含
        self.assertIn("<video>", msg)  # 断言包含

    # TestGenerateEmbeddingConvs类的测试nonetext
    def test_none_text(self):
        """Test embedding conv with None text (only media)."""
        convs = generate_embedding_convs(
            texts=[None],
            images=["http://example.com/img.jpg"],
            videos=[None],
            template_name="chatml",
        )
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)  # 断言包含
        # None text should not produce "None" string
        self.assertNotIn("None", msg)  # 断言不包含

    # TestGenerateEmbeddingConvs类的测试multipleitems
    def test_multiple_items(self):
        """Test generating multiple embedding conversations."""
        convs = generate_embedding_convs(
            texts=["text1", "text2"],
            images=[None, None],
            videos=[None, None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 2)  # 断言相等


# TestGetFullMultimodalTextPrompt类
class TestGetFullMultimodalTextPrompt(CustomTestCase):

    # TestGetFullMultimodalTextPrompt类的测试addsmissingimagetokens
    def test_adds_missing_image_tokens(self):
        """Test adding missing image tokens to prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 3, "Describe this.")
        self.assertEqual(result.count("<image>"), 3)  # 断言相等
        self.assertIn("Describe this.", result)  # 断言包含

    # TestGetFullMultimodalTextPrompt类的测试preservesexistingtokens
    def test_preserves_existing_tokens(self):
        """Test that existing tokens in prompt are preserved."""
        result = _get_full_multimodal_text_prompt(
            "<image>", 2, "<image> What about this?"
        )
        self.assertEqual(result.count("<image>"), 2)  # 断言相等

    # TestGetFullMultimodalTextPrompt类的测试alltokenspresentnoaddition
    def test_all_tokens_present_no_addition(self):
        """Test no addition when all tokens are already present."""
        result = _get_full_multimodal_text_prompt("<image>", 2, "<image> and <image>")
        self.assertEqual(result, "<image> and <image>")  # 断言相等

    # TestGetFullMultimodalTextPrompt类的测试moretokensthandataraises
    def test_more_tokens_than_data_raises(self):
        """Test that more placeholders than data items raises ValueError."""
        with self.assertRaises(ValueError):  # 断言抛出异常
            _get_full_multimodal_text_prompt("<image>", 1, "<image> <image>")

    # TestGetFullMultimodalTextPrompt类的测试zerocountwithnotokens
    def test_zero_count_with_no_tokens(self):
        """Test zero modality count with no tokens in prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 0, "Just text")
        self.assertEqual(result, "Just text")  # 断言相等

    # TestGetFullMultimodalTextPrompt类的测试videotokens
    def test_video_tokens(self):
        """Test adding missing video tokens."""
        result = _get_full_multimodal_text_prompt("<video>", 2, "Describe:")
        self.assertEqual(result.count("<video>"), 2)  # 断言相等
        self.assertIn("Describe:", result)  # 断言包含

    # TestGetFullMultimodalTextPrompt类的测试tokensjoinedwithnewline
    def test_tokens_joined_with_newline(self):
        """Test that missing tokens are joined with newlines before prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 3, "text")
        # 3 images, 0 in prompt → 3 added, joined by \n, then \n before text
        lines = result.split("\n")
        self.assertEqual(lines[0], "<image>")  # 断言相等
        self.assertEqual(lines[1], "<image>")  # 断言相等
        self.assertEqual(lines[2], "<image>")  # 断言相等
        self.assertEqual(lines[3], "text")  # 断言相等


# TestGenerateChatConv类
class TestGenerateChatConv(CustomTestCase):
    """Test generate_chat_conv with real Pydantic message objects."""

    def _make_request(self, messages):
        """Create a real ChatCompletionRequest with given messages."""
        return ChatCompletionRequest(messages=messages, model="test")

    # TestGenerateChatConv类的测试simpleusermessage
    def test_simple_user_message(self):
        """Test basic user string message."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hello")]
        )
        conv = generate_chat_conv(request, "chatml")
        # user message + blank assistant placeholder
        self.assertEqual(len(conv.messages), 2)  # 断言相等
        self.assertIn("Hello", conv.messages[0][1])  # 断言包含
        self.assertIsNone(conv.messages[1][1])  # 断言为None

    # TestGenerateChatConv类的测试systemthenuser
    def test_system_then_user(self):
        """Test system message followed by user message."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(role="system", content="Be helpful"),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.system_message, "Be helpful")  # 断言相等
        self.assertIn("Hi", conv.messages[0][1])  # 断言包含

    # TestGenerateChatConv类的测试systemmessageaslist
    def test_system_message_as_list(self):
        """Test system message given as a single-element list of text parts."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(
                    role="system",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="System text"
                        )
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.system_message, "System text")  # 断言相等

    # TestGenerateChatConv类的测试systemmessageinvalidlistraises
    def test_system_message_invalid_list_raises(self):
        """Test that system message with non-text content raises ValueError."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(
                    role="system",
                    content=[
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        )
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        with self.assertRaises(ValueError):  # 断言抛出异常
            generate_chat_conv(request, "chatml")

    # TestGenerateChatConv类的测试multiturnconversation
    def test_multi_turn_conversation(self):
        """Test multi-turn user/assistant conversation."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="What is 2+2?"),
                ChatCompletionMessageGenericParam(role="assistant", content="4"),
                ChatCompletionMessageUserParam(role="user", content="And 3+3?"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        # 3 explicit messages + 1 blank assistant placeholder
        self.assertEqual(len(conv.messages), 4)  # 断言相等
        self.assertEqual(conv.messages[1][1], "4")  # 断言相等
        self.assertIsNone(conv.messages[3][1])  # 断言为None

    # TestGenerateChatConv类的测试assistantmessageaslist
    def test_assistant_message_as_list(self):
        """Test assistant message given as a single-element list of text parts."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="Hi"),
                ChatCompletionMessageGenericParam(
                    role="assistant",
                    content=[
                        ChatCompletionMessageContentTextPart(type="text", text="Hello!")
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="How are you?"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.messages[1][1], "Hello!")  # 断言相等

    # TestGenerateChatConv类的测试assistantinvalidlistraises
    def test_assistant_invalid_list_raises(self):
        """Test that assistant message with non-text content raises ValueError."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="Hi"),
                ChatCompletionMessageGenericParam(
                    role="assistant",
                    content=[
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        )
                    ],
                ),
            ]
        )
        with self.assertRaises(ValueError):  # 断言抛出异常
            generate_chat_conv(request, "chatml")

    # TestGenerateChatConv类的测试stringmessagesraises
    def test_string_messages_raises(self):
        """Test that passing messages as a raw string raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        # Manually override messages to be a string to trigger validation
        request.__dict__["messages"] = "not a list"
        with self.assertRaises(ValueError):  # 断言抛出异常
            generate_chat_conv(request, "chatml")

    # TestGenerateChatConv类的测试usermessagewithimage
    def test_user_message_with_image(self):
        """Test user message with image content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="What's in this image?"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/cat.jpg"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.image_data), 1)  # 断言相等
        self.assertEqual(conv.image_data[0].url, "http://example.com/cat.jpg")  # 断言相等
        msg = conv.messages[0][1]
        self.assertIn("What's in this image?", msg)  # 断言包含

    # TestGenerateChatConv类的测试usermessagewithvideo
    def test_user_message_with_video(self):
        """Test user message with video content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Describe this video"
                        ),
                        ChatCompletionMessageContentVideoPart(
                            type="video_url",
                            video_url=ChatCompletionMessageContentVideoURL(
                                url="http://example.com/vid.mp4"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.video_data), 1)  # 断言相等
        self.assertEqual(conv.video_data[0], "http://example.com/vid.mp4")  # 断言相等

    # TestGenerateChatConv类的测试usermessagewithaudio
    def test_user_message_with_audio(self):
        """Test user message with audio content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Transcribe this"
                        ),
                        ChatCompletionMessageContentAudioPart(
                            type="audio_url",
                            audio_url=ChatCompletionMessageContentAudioURL(
                                url="http://example.com/audio.wav"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.audio_data), 1)  # 断言相等
        self.assertEqual(conv.audio_data[0], "http://example.com/audio.wav")  # 断言相等

    # TestGenerateChatConv类的测试usermessageimageatprefix
    def test_user_message_image_at_prefix(self):
        """Test image_token_at_prefix=True puts image token before text."""
        # Register a temporary template with image_token_at_prefix=True
        tmp_name = "_test_prefix_img"
        register_conv_template(
            Conversation(
                name=tmp_name,
                roles=("<|im_start|>user", "<|im_start|>assistant"),
                messages=[],
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>",
                image_token_at_prefix=True,
            )
        )
        try:
            request = self._make_request(
                [
                    ChatCompletionMessageUserParam(
                        role="user",
                        content=[
                            ChatCompletionMessageContentTextPart(
                                type="text", text="Describe"
                            ),
                            ChatCompletionMessageContentImagePart(
                                type="image_url",
                                image_url=ChatCompletionMessageContentImageURL(
                                    url="http://example.com/img.jpg"
                                ),
                            ),
                        ],
                    )
                ]
            )
            conv = generate_chat_conv(request, tmp_name)
            msg = conv.messages[0][1]
            # Image token should be BEFORE "Describe"
            img_pos = msg.find("<image>")
            txt_pos = msg.find("Describe")
            self.assertGreater(txt_pos, img_pos)  # 断言大于
        finally:
            del chat_templates[tmp_name]

    # TestGenerateChatConv类的测试deepseekvl2modalitysupplement
    def test_deepseek_vl2_modality_supplement(self):
        """Test deepseek-vl2 modality supplement (add_token_as_needed path)."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Describe both"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img1.jpg"
                            ),
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img2.jpg"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "deepseek-vl2")
        self.assertEqual(len(conv.image_data), 2)  # 断言相等
        msg = conv.messages[0][1]
        # deepseek-vl2 uses _get_full_multimodal_text_prompt to add image tokens
        self.assertIn("Describe both", msg)  # 断言包含

    # TestGenerateChatConv类的测试unknownroleraises
    def test_unknown_role_raises(self):
        """Test that an unknown message role raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        # Manually inject a message with unknown role
        from types import SimpleNamespace

        request.__dict__["messages"] = [SimpleNamespace(role="alien", content="Hi")]
        with self.assertRaises(ValueError):  # 断言抛出异常
            generate_chat_conv(request, "chatml")

    # TestGenerateChatConv类的测试usermessagemanyimagesaddsnewline
    def test_user_message_many_images_adds_newline(self):
        """Test that >16 images triggers newline before text content."""
        image_parts = [
            ChatCompletionMessageContentImagePart(
                type="image_url",
                image_url=ChatCompletionMessageContentImageURL(
                    url=f"http://example.com/img{i}.jpg"
                ),
            )
            for i in range(17)
        ]
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="Describe all")
        ] + image_parts
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content=content)]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.image_data), 17)  # 断言相等
        # With >16 images, text content is prefixed with "\n"
        self.assertIn("\nDescribe all", conv.messages[0][1])  # 断言包含


# TestGetModelType类
class TestGetModelType(CustomTestCase):

    # TestGetModelType类的测试nonexistentpathreturnsnone
    def test_nonexistent_path_returns_none(self):
        """Test that a path without config.json returns None."""
        result = get_model_type("/nonexistent/path/abc123")
        self.assertIsNone(result)  # 断言为None

    # TestGetModelType类的测试validconfigreturnsmodeltype
    def test_valid_config_returns_model_type(self):
        """Test reading model_type from a real config.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model_type": "llama", "hidden_size": 4096}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            result = get_model_type(tmpdir)
            self.assertEqual(result, "llama")  # 断言相等

    # TestGetModelType类的测试configwithoutmodeltypereturnsnone
    def test_config_without_model_type_returns_none(self):
        """Test that config.json without model_type key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"hidden_size": 4096}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            result = get_model_type(tmpdir)
            self.assertIsNone(result)  # 断言为None

    # TestGetModelType类的测试invalidjsonreturnsnone
    def test_invalid_json_returns_none(self):
        """Test that malformed config.json returns None (JSONDecodeError)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("not valid json{{{")
            result = get_model_type(tmpdir)
            self.assertIsNone(result)  # 断言为None


if __name__ == "__main__":
    unittest.main()
