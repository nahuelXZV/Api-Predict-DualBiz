from rest_framework import serializers


class ModelMetadataSerializer(serializers.Serializer):
    name = serializers.CharField()
    version = serializers.CharField()
    feature_names = serializers.ListField(child=serializers.CharField())
    target_name = serializers.CharField()
    hyperparams = serializers.DictField()
    loaded_at = serializers.DateTimeField()
    trained_at = serializers.DateTimeField(allow_null=True)
    extra = serializers.DictField()
    path_model = serializers.CharField()
