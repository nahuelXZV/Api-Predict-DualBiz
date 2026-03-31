from rest_framework import serializers


class PredictResponseSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    predictions = serializers.JSONField()
    success = serializers.BooleanField()
