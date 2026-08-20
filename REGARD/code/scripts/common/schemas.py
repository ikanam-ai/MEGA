"""Pydantic-схемы записей, которыми обмениваются этапы пайплайна."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TargetRecord(BaseModel):
    target_id: str
    target_name: str
    target_family: str
    description_en: str | None = None
    geography: str | None = None
    cis_relevance_note: str | None = None


class GenerationRecord(BaseModel):
    target_id: str
    target_name: str
    target_family: str
    model_id: str
    model_role: Literal["local", "international"]
    prompt_id: str
    prompt_text: str
    response_text: str
    response_word_len: int
    finish_reason: str
    generation_id: str
    run_id: str
    timestamp: str
    language: str = "ru"


class JudgeScore(BaseModel):
    """Унифицированный результат вызова judge-провайдера."""

    lm_valence: float = Field(ge=0, le=1)
    lm_arousal: float = Field(ge=0, le=1)
    lm_dominance: float = Field(ge=0, le=1)
    lm_confidence: float = Field(ge=0, le=1)
    target_coverage: float = Field(ge=0, le=1)
    knowledge_specificity: float = Field(ge=0, le=1)
    refusal_flag: bool
    generic_answer_flag: bool
    unknown_flag: bool
    hallucination_or_mismatch_flag: bool
    valid_for_vad_analysis: bool
    evidence_spans: list[str]
    rationale: str


class JudgeScoreRecord(BaseModel):
    generation_id: str
    target_id: str
    model_id: str
    prompt_id: str
    prompt_text: str
    judge_model: str
    lm_valence: float
    lm_arousal: float
    lm_dominance: float
    lm_confidence: float
    target_coverage: float
    knowledge_specificity: float
    refusal_flag: bool
    generic_answer_flag: bool
    unknown_flag: bool
    hallucination_or_mismatch_flag: bool
    valid_for_vad_analysis: bool
    evidence_spans: list[str]
    rationale: str
    scoring_contract: str = "td_v1_contract_ru"


class AnnotationBatchItem(BaseModel):
    annotation_item_id: str
    target_id: str
    target_name: str
    local_model_id: str
    local_response_text: str
    intl_model_id: str
    intl_response_text: str
    assigned_annotators: list[str]
