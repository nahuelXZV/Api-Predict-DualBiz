from rest_framework import serializers


class TrainResponseSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    version = serializers.CharField()
    steps_executed = serializers.ListField(child=serializers.CharField())
    errors = serializers.ListField(child=serializers.CharField())
    success = serializers.BooleanField()
