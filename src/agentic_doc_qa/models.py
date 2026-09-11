#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_model(model_name: str, base_url: str, api_key: str) -> OpenAIChatModel:
    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
