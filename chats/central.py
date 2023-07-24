from chats import stablelm
from chats import alpaca
from chats import koalpaca
from chats import flan_alpaca
from chats import os_stablelm
from chats import vicuna
from chats import stable_vicuna
from chats import starchat
from chats import wizard_coder
from chats import redpajama
from chats import mpt
from chats import alpacoom
from chats import baize
from chats import guanaco
from chats import falcon
from chats import wizard_falcon
from chats import xgen
from chats import llama2
from chats import freewilly
from chats import custom

def chat_stream(
    idx, local_data, user_message, state, 
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid,
    internet_option, serper_api_key
):
    model_type = state["model_type"]

    internet_option = internet_option == "on" and serper_api_key.strip() != ""
    if model_type == "alpacoom":
        cs = alpacoom.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "baize":
        cs = baize.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "custom":
        cs = custom.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "falcon":
        cs = falcon.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "flan-alpaca":
        cs = flan_alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "free-willy":
        cs = freewilly.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "guanaco":
        cs = guanaco.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type in ["koalpaca-polyglot", "kullm-polyglot"]:
        cs = koalpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "llama2":
        cs = llama2.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "mpt":
        cs = mpt.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "os-stablelm":
        cs = os_stablelm.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type in ["redpajama", "redpajama-instruct"]:
        cs = redpajama.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "stable-vicuna":
        cs = stable_vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "stablelm":
        cs = stablelm.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "starchat":
        cs = starchat.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type in [
        "t5-vicuna",
        "vicuna",
        "wizardlm",
        "wizard-vicuna",
        "airoboros",
        "samantha-vicuna",
        "evolinstruct-vicuna",
    ]:
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type in [
        "upstage-llama",
        "alpaca",
        "openllama",
        "orcamini",
        "alpaca-gpt4",
        "nous-hermes",
        "replit-instruct",
        "llama-deus",
        "camel",
        "lazarus",
        "chronos",
    ]:
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "wizard-coder":
        cs = wizard_coder.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "wizard-falcon":
        cs = wizard_falcon.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "xgen":
        cs = xgen.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    yield from cs        
        