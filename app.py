import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import io

import plotly.express as px
import plotly.graph_objects as go

import folium
from folium.plugins import AntPath

# Si usas base64 en tu código (no estaba en los callbacks que revisamos, pero lo dejo por si acaso):
# import base64 

# Mapeo de niveles a colores hexadecimales para mayor claridad
COLORES_CONGESTION = {
    1: "#F0E850",  # Amarillo
    2: '#FF9800',  # Naranja
    3: '#F44336',  # Rojo
    4: "#8A1414"   # Rojo Opaco/Oscuro
}

max_diff_temporal = 10 #minutes
# Cargar tu dataframe (asume que 'geometry' es un string WKT)
df_join = pd.read_csv('congestiones_con_grupos.csv')

# Convertir hora_extraccion a formato datetime y extraer solo la hora como string para el dropdown
df_join['hora_extraccion_dt'] = pd.to_datetime(df_join['hora_extraccion_local'])
# Usaremos un formato HH:MM para el dropdown
df_join['hora_formateada'] = df_join['hora_extraccion_dt'].dt.strftime('%H:%M')

# Preparar opciones del primer dropdown (grupos)
conteo_por_grupo = df_join.groupby('congestion_group_id')['id'].count().reset_index(name='count')
opciones_dropdown_grupo = []
for index, row in conteo_por_grupo.iterrows():
    label_texto = f"Grupo {row['congestion_group_id']} ({row['count']} registros)"
    valor_grupo = row['congestion_group_id']
    opciones_dropdown_grupo.append({'label': label_texto, 'value': valor_grupo})

# Preparar opciones del segundo dropdown (horas únicas disponibles en todo el dataset)
horas_unicas = sorted(df_join['hora_formateada'].unique())
opciones_dropdown_hora = [{'label': h, 'value': h} for h in horas_unicas]


def crear_simbologia_congestiones():
    """
    Genera un componente HTML para mostrar la leyenda de niveles de congestión.
    Incluye título y descripciones completas.
    """
    items_leyenda = []
    
    # Preparamos las descripciones completas en el bucle
    descripciones = {
        1: 'Congestión Nivel 1: Bajo (Fluido)',
        2: 'Congestión Nivel 2: Moderado',
        3: 'Congestión Nivel 3: Alto',
        4: 'Congestión Nivel 4: Extremo'
    }

    # Iteramos sobre los niveles y colores
    colores_map = COLORES_CONGESTION
    
    for nivel in sorted(colores_map.keys()):
        color = colores_map[nivel]
        descripcion = descripciones[nivel]

        items_leyenda.append(
            html.Div([
                # Cuadro de color
                html.Span(
                    style={
                        'backgroundColor': color,
                        'height': '15px',
                        'width': '15px',
                        'display': 'inline-block',
                        'marginRight': '10px',
                        'border': '1px solid #ccc'
                    }
                ),
                # El texto descriptivo
                html.Span(descripcion)
            ], style={'marginBottom': '5px'})
        )
    
    return html.Div(
        [
            # Título añadido
            html.H4("Simbología de Congestión", style={'marginBottom': '15px', 'fontWeight': 'bold'}),
            # Contenedor de los items de la leyenda
            *items_leyenda
        ],
        style={
            'border': '1px solid #ddd',
            'padding': '15px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'marginTop': '20px',
            'width': '300px', # Se ajusta el ancho para que quepa el texto más largo
            'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'
        }
    )


# Inicializar la app
# Define el meta tag que apunta a tu archivo dentro de la carpeta assets/
# meta_tags = [
#     {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
#     # Referencia al archivo en assets/ (Dash se encarga de la ruta correcta)
#     {"rel": "icon", "href": "/assets/icono_auto.png", "type": "image/png"} 
# ]

# Inicializa la app con el título y los meta tags
app = dash.Dash(
    __name__, 
    title="Congestiones de Tráfico", 
    #meta_tags=meta_tags,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

menu_items = [
    dbc.NavItem(dbc.NavLink("Dashboard Principal", href="/")),
    dbc.DropdownMenu(
        children=[
            dbc.DropdownMenuItem("Análisis de Tráfico", href="/analisis-trafico"),
            dbc.DropdownMenuItem("Datos Crudos (Tabla)", href="/datos-crudos"),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem("Acerca de...", href="/acerca"),
        ],
        nav=True,
        in_navbar=True,
        label="Navegación",
    ),
    dbc.NavItem(dbc.NavLink("Contacto", href="/contacto")),
]

# Barra de Navegación Profesional (Header)
navbar = dbc.NavbarSimple(
    children=menu_items,
    brand="Congestiones de Tráfico 🚗",
    brand_href="#",
    color="dark", # Color de fondo oscuro
    dark=True,    # Texto blanco
    sticky="top", # Se mantiene fijo en la parte superior
    className="shadow-sm" # Añade una sombra sutil
)


app.layout = dbc.Container([
    navbar,
    
    # Controles de Selección de Grupo y Simbología en una sola fila
    dbc.Row([
        dbc.Col([
            html.Label("Seleccionar Grupo de Estudio:", className="fw-bold mb-1"),
            dcc.Dropdown(
                id='grupo-dropdown',
                options=opciones_dropdown_grupo, 
                value=opciones_dropdown_grupo[0]['value'], 
                clearable=False,
                className="dbc" # Aplica estilo Bootstrap al dropdown
            )
        ], md=5),

        # Mueve la simbología a la derecha de los controles principales
        dbc.Col([
            crear_simbologia_congestiones() # Asume que esta función devuelve un Div estilizado
        ], md=7, className="d-flex justify-content-end") # Alinea a la derecha
    ], className="mb-4 align-items-center"),

    # CONTENEDOR PARA EL GRAFICO DE EVOLUCION
    # Usa un espaciado mejorado y un título claro
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Evolución Temporal de la Longitud de Congestión", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='grafico-evolucion')
                ])
            ], className="shadow-sm") # Añade una sombra sutil a las tarjetas
        ])
    ], className="mb-4"),
            
    # Controles de Hora para Mapas Duales
    html.H4("Comparación de Mapas por Hora", className="mt-4 mb-3 text-muted border-bottom pb-2"),

    dbc.Row([
        # PRIMER DROPDOWN DE HORA (A)
        dbc.Col([
            html.Label("Hora Inicial (A):", className="fw-bold mb-1"),
            dcc.Dropdown(
                id='hora-dropdown-A',
                options=opciones_dropdown_hora,
                value=horas_unicas[0],
                clearable=False,
                className="dbc"
            )
        ], md=3),

        # SEGUNDO DROPDOWN DE HORA (B)
        dbc.Col([
            html.Label("Hora Final (B):", className="fw-bold mb-1"),
            dcc.Dropdown(
                id='hora-dropdown-B',
                options=opciones_dropdown_hora,
                value=horas_unicas[-1],
                clearable=False,
                className="dbc"
            )
        ], md=3),
    ], className="mb-4"),


    # CONTENEDORES DUALES PARA LOS MAPAS (Con colores neutrales)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Usa bg-dark para un toque de color oscuro y texto blanco
                dbc.CardHeader(html.H6("Mapa de Congestión: Hora A", className="text-white mb-0 bg-dark py-2 px-3")),
                dbc.CardBody([
                    html.Iframe(id='mapa-iframe-A', srcDoc=None, width='100%', height='800px')
                ], className="p-0")
            ], className="shadow-sm")
        ], md=6),
        dbc.Col([
            dbc.Card([
                 # Usa bg-white para un fondo blanco y texto oscuro
                 dbc.CardHeader(html.H6("Mapa de Congestión: Hora B", className="text-white mb-0 bg-dark py-2 px-3")),
                dbc.CardBody([
                    html.Iframe(id='mapa-iframe-B', srcDoc=None, width='100%', height='800px')
                ], className="p-0")
            ], className="shadow-sm")
        ], md=6),
    ], className="mb-4"),

    # CONTENEDOR PARA LA TABLA
    html.H4("Detalle de Registros", className="mt-4 mb-3 text-muted border-bottom pb-2"),
    dbc.Row([
        dbc.Col([
            html.Div(id='tabla-container')
        ])
    ]),

    # Footer simple
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Waze Traffic Data Analysis | Proyecto GCP & Dash Python", className="text-center text-muted small")
        ])
    ])
    
], fluid=True, className="bg-light py-4") # Usa un fondo gris claro para que las tarjetas resalten
# --- Funciones Auxiliares para generar mapas y aplicar AntPath ---

def generar_mapa_folium(df_data, hora_label):
    if df_data.empty:
        return "<div>No hay datos disponibles para esta hora y grupo.</div>"

    # Convertir a GeoDataFrame
    df_data['geometry_obj'] = df_data['geometry'].apply(wkt.loads)
    gdf_mapa = gpd.GeoDataFrame(df_data, geometry='geometry_obj', crs="EPSG:4326")

    # Limpiar dtypes para evitar errores de JSON si Folium los necesita internamente
    for col in gdf_mapa.columns:
        if gdf_mapa[col].dtype == 'datetime64[ns]' or gdf_mapa[col].dtype == 'object':
            if col not in ['geometry_obj', 'geometry']: # Excluimos los objetos shapely
                gdf_mapa[col] = gdf_mapa[col].astype(str)

    centro_mapa = gdf_mapa.unary_union.centroid

    # --- CAMBIO AQUÍ: Usar 'CartoDB Dark Matter' para un tema oscuro ---
    m = folium.Map(
        location=[centro_mapa.y, centro_mapa.x], 
        zoom_start=13, 
        tiles="OpenStreetMap" # <-- Estilo de mapa negro
        # También puedes usar "OpenStreetMap" (el default) o "Stamen Toner"
    )

    # Iterar y añadir AntPaths
    for _, row in gdf_mapa.iterrows():
        geom = row['geometry_obj']
        
        if geom.geom_type in ['LineString', 'MultiLineString']:
            # Extraer coordenadas en formato [lat, lon]
            coords = []
            if geom.geom_type == 'LineString':
                coords = [list(c[::-1]) for c in geom.coords] # Invertir lon, lat a lat, lon
            elif geom.geom_type == 'MultiLineString':
                 for line in geom.geoms:
                    coords.extend([list(c[::-1]) for c in line.coords])



            # Dentro de tu bucle o función que procesa cada fila (row):
            velocidad = row['speedKMH']
            nivel_congestion = row['level']
            color_linea = COLORES_CONGESTION.get(nivel_congestion, '#9E9E9E') # El gris es un color por defecto si el nivel no es válido

            AntPath(
                locations=coords,
                delay=2000,
                color=color_linea,
                pulse_color='white',
                weight=5,
                opacity=0.8,
                tooltip=f"Calle: {row['street']}<br>Velocidad: {velocidad:.1f} km/h,<br> Nivel de Congestión: {nivel_congestion}"
            ).add_to(m)

    # Devolver HTML del mapa
    data = io.BytesIO()
    m.save(data, close_file=False)
    return data.getvalue().decode('utf-8')

@app.callback(
    Output('tabla-container', 'children'),
    Output('grafico-evolucion', 'figure'),
    Input('grupo-dropdown', 'value')
)
def actualizar_contenido_grupo(grupo_seleccionado):
    df_filtrado = df_join[df_join['congestion_group_id'] == grupo_seleccionado].copy()
    
    # --- LIMPIEZA Y ORDENAMIENTO ---
    df_filtrado['level'] = df_filtrado['level'].fillna(0).astype(int)
    df_filtrado['street'] = df_filtrado['street'].fillna('Sin Calle').astype(str)
    df_filtrado['hora_extraccion_dt'] = pd.to_datetime(df_filtrado['hora_extraccion_local'])
    df_filtrado = df_filtrado.sort_values(by='hora_extraccion_dt')
    
    if df_filtrado.empty:
        return html.Div("No hay datos para mostrar"), go.Figure()

    fig = go.Figure()

    # --- DIBUJO DE CALLES (CAPAS SUPERPUESTAS) ---
    df_siguiente_punto = df_filtrado.shift(-1)
    
    for i in range(len(df_filtrado) - 1):
        punto_actual = df_filtrado.iloc[i]
        punto_siguiente = df_siguiente_punto.iloc[i]
        
        time_diff = punto_siguiente['hora_extraccion_dt'] - punto_actual['hora_extraccion_dt']
        if time_diff > pd.Timedelta(minutes=max_diff_temporal):
            continue 
            
        nivel = punto_actual['level']
        color_segmento = COLORES_CONGESTION.get(nivel, '#9E9E9E')
        x_coords = [punto_actual['hora_extraccion_dt'], punto_siguiente['hora_extraccion_dt']]
        y_coords = [punto_actual['length'], punto_siguiente['length']]
        
        # 1. CAPA INFERIOR: Borde de la calle (Casing negro)
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color='#2c3e50', width=10), # Más ancha y oscura
            hoverinfo='skip', showlegend=False
        ))
        
        # 2. CAPA MEDIA: Color de congestión
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color=color_segmento, width=16), # Color principal
            hoverinfo='skip', showlegend=False
        ))

        # 3. CAPA SUPERIOR: Línea central de carriles (Discontinua)
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color='white', width=4, dash='dot'), # Línea punteada blanca
            hoverinfo='skip', showlegend=False
        ))
        
        # --- LÓGICA DEL AUTO (SOLO INICIO) ---
        es_inicio = (i == 0) or (punto_actual['hora_extraccion_dt'] - df_filtrado.iloc[i-1]['hora_extraccion_dt'] > pd.Timedelta(minutes=max_diff_temporal))
        if es_inicio:
            fig.add_trace(go.Scatter(
                x=[punto_actual['hora_extraccion_dt']], y=[punto_actual['length']],
                mode='text', text="🚗", textposition="top center",
                textfont=dict(size=24), hoverinfo='skip', showlegend=False
            ))

    # 4. CAPA DE MARCADORES (Puntos con Hover)
    all_hover = df_filtrado.apply(lambda row: f"Calle: {row['street']}<br>Nivel: {row['level']}<br>Velocidad: {row['speedKMH']:.1f} km/h<br>Hora: {row['hora_extraccion_local']}", axis=1)
    
    fig.add_trace(go.Scatter(
        x=df_filtrado['hora_extraccion_dt'], y=df_filtrado['length'],
        mode='markers',
        marker=dict(size=9, color='white', line=dict(width=2, color='#2c3e50')),
        hoverinfo='text', text=all_hover, showlegend=False
    ))

    fig.update_layout(
        title=f"Evolución de Congestión - Grupo {grupo_seleccionado}",
        xaxis_title="Tiempo", yaxis_title="Longitud (m)",
        template="ggplot2", # Cambiado a oscuro para resaltar el efecto calle
        showlegend=False,
        hovermode='closest'
    )

    # --- 2. Preparar datos para la tabla (código final con formato de floats) ---
    df_display = df_join[df_join['congestion_group_id'] == grupo_seleccionado].copy()

    # Manejo de strings seguro (ya lo tenías):
    df_display['street'] = df_display['street'].fillna('Sin Calle').apply(lambda x: str(x)[:30] + '...' if len(str(x)) > 30 else str(x))
    df_display['city'] = df_display['city'].fillna('Sin Ciudad').apply(lambda x: str(x)[:20] + '...' if len(str(x)) > 20 else x)

    # >>>>> FORMATO DE NÚMEROS FLOTANTES A 2 DECIMALES:
    # Identificamos las columnas que probablemente sean floats y las formateamos como strings
    df_display['speed'] = df_display['speed'].round(2).astype(str)
    df_display['speedKMH'] = df_display['speedKMH'].round(2).astype(str)
    df_display['length'] = df_display['length'].round(2).astype(str)
    # Si 'first_two_coords' tiene floats, también podrías necesitar formatearlos si es necesario.

    # Definición de las columnas finales de la tabla:
    columnas_tabla = ['country', 'level', 'city', 'id', 'speed',
                      'speedKMH', 'street', 'length', 'first_two_coords', 
                      'dia_extraccion', 'hora_extraccion_local']

    tabla = dbc.Table.from_dataframe(
        df_display[columnas_tabla].sort_values(by='hora_extraccion_local'),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )

    tabla_card = dbc.Card([
        dbc.CardBody([
            html.H5(f"Total de registros: {len(df_filtrado)}", className="mb-3"),
            html.Div(tabla, style={'overflowX': 'auto'})
        ])
    ])

    return tabla_card, fig


# SEGUNDO CALLBACK (NUEVO): Actualiza los mapas duales
@app.callback(
    Output('mapa-iframe-A', 'srcDoc'),
    Output('mapa-iframe-B', 'srcDoc'),
    Input('grupo-dropdown', 'value'),
    Input('hora-dropdown-A', 'value'),
    Input('hora-dropdown-B', 'value')
)
def actualizar_mapas_duales(grupo_seleccionado, hora_A, hora_B):
    # Filtrar datos para la Hora A
    df_A = df_join[
        (df_join['congestion_group_id'] == grupo_seleccionado) &
        (df_join['hora_formateada'] == hora_A)
    ].copy()
    
    # Filtrar datos para la Hora B
    df_B = df_join[
        (df_join['congestion_group_id'] == grupo_seleccionado) &
        (df_join['hora_formateada'] == hora_B)
    ].copy()

    # Generar ambos mapas usando la función auxiliar
    mapa_html_A = generar_mapa_folium(df_A, hora_A)
    mapa_html_B = generar_mapa_folium(df_B, hora_B)
    
    return mapa_html_A, mapa_html_B


if __name__ == '__main__':
    app.run(debug=True)
