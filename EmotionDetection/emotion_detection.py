"""Format emotion predictions from the Watson NLP service."""

import requests


def emotion_detector(text_to_analyze: str) -> dict[str, float | str]:
    """Return five emotion scores and the emotion with the highest score."""
    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    prediction = response.json()["emotionPredictions"][0]["emotion"]
    scores = {
        emotion: prediction[emotion]
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    }
    dominant_emotion = max(scores, key=scores.__getitem__)
    return {**scores, "dominant_emotion": dominant_emotion}
