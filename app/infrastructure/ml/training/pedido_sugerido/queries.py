CONSULTA_BASE = """
    SELECT
        FechaVenta,
        ID_Ruta,
        ID_Zona,
        ID_Cliente,
        Producto,
        CantidadVendida,
        LineaProducto,
        Marca,
        ClasificacionCliente,
        Nombre_Ruta,
        Nombre_Zona,
        Sucursal,
        Vendedor
    FROM dbo.VentasHistoricas
"""
