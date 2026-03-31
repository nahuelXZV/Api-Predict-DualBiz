from rest_framework import serializers


class TrainRequestSerializer(serializers.Serializer):
    model_name = serializers.CharField(
        default="pedido_sugerido",
        help_text="Nombre del modelo a entrenar.",
    )
    version = serializers.CharField(
        default="1.0",
        help_text="Versión del modelo.",
    )
