import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import requests

st.set_page_config(layout="wide")
st.title("Matriz de Correlación de Acciones")

# Data source functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error al descargar datos de yfinance para {ticker}: {e}")
        return pd.DataFrame()

def calculate_ticker_ratio(data1, data2):
    """Calculate the ratio between two tickers' closing prices."""
    if data1.empty or data2.empty:
        raise ValueError("Uno o ambos conjuntos de datos están vacíos")

    # Ensure both datasets have the same index
    common_dates = data1.index.intersection(data2.index)
    if len(common_dates) == 0:
        raise ValueError("No hay fechas superpuestas entre los dos tickers")

    data1 = data1.reindex(common_dates)
    data2 = data2.reindex(common_dates)

    # Calculate ratio
    if isinstance(data1.columns, pd.MultiIndex):
        close1 = data1['Close'].iloc[:, 0]
    else:
        close1 = data1['Close']

    if isinstance(data2.columns, pd.MultiIndex):
        close2 = data2['Close'].iloc[:, 0]
    else:
        close2 = data2['Close']

    ratio = pd.DataFrame({'Close': close1 / close2}, index=common_dates)
    return ratio

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        # Rest of the function remains the same...

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c', 'o', 'h', 'l', 'v']):
                st.error(f"Datos incompletos recibidos para {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c'],
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Volume': data['v']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df[['Close']]  # Return only Close column for consistency
        else:
            st.error(f"Error al obtener datos para {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error al descargar datos de analisistecnico para {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                st.error(f"Error en la respuesta de la API para {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['bars'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df['Close'] = df['close']
            df = df[['Date', 'Close']]
            df.set_index('Date', inplace=True)
            df = df.sort_index().drop_duplicates()
            return df
        else:
            st.error(f"Error al obtener datos para {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error al descargar datos de IOL para {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        # Remove .BA and add 24HS for BYMA format
        symbol = ticker.replace('.BA', '')
        if not symbol.endswith(' 24HS'):
            symbol = f"{symbol} 24HS"

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c']):
                st.error(f"Datos incompletos recibidos para {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            st.error(f"Error al obtener datos para {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error al descargar datos de ByMA Data para {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_stock_data(ticker, start_date, end_date, source='YFinance'):
    try:
        if source == 'YFinance':
            return descargar_datos_yfinance(ticker, start_date, end_date)
        elif source == 'AnálisisTécnico.com.ar':
            return descargar_datos_analisistecnico(ticker, start_date, end_date)
        elif source == 'IOL (Invertir Online)':
            return descargar_datos_iol(ticker, start_date, end_date)
        elif source == 'ByMA Data':
            return descargar_datos_byma(ticker, start_date, end_date)
        else:
            st.error(f"Fuente de datos desconocida: {source}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker} desde {source}: {e}")
        return pd.DataFrame()

def prepare_correlation_data(tickers, start_date, end_date, source, ma_periods):
    all_data = pd.DataFrame()
    for ticker in tickers:
        if '/' in ticker:
            # Handle ratio
            ticker1, ticker2 = [t.strip() for t in ticker.split('/')]
            data1 = fetch_stock_data(ticker1, start_date, end_date, source)
            data2 = fetch_stock_data(ticker2, start_date, end_date, source)
            if not data1.empty and not data2.empty:
                stock_data = calculate_ticker_ratio(data1, data2)
                if ma_periods > 1:
                    all_data[ticker] = stock_data['Close'].rolling(window=ma_periods).mean()
                else:
                    all_data[ticker] = stock_data['Close']
            else:
                st.error(f"No se pudieron obtener datos para la relación {ticker}. Saltando.")
        else:
            # Handle single ticker
            stock_data = fetch_stock_data(ticker, start_date, end_date, source)
            if not stock_data.empty:
                if isinstance(stock_data.columns, pd.MultiIndex):
                    close_prices = stock_data['Close'].iloc[:, 0]
                else:
                    close_prices = stock_data['Close']
                if ma_periods > 1:
                    all_data[ticker] = close_prices.rolling(window=ma_periods).mean()
                else:
                    all_data[ticker] = close_prices
            else:
                st.error(f"No se pudieron obtener datos para {ticker}. Saltando.")
    return all_data

def plot_correlation_matrix(data, title, method='pearson'):
    # Clear any existing plots
    plt.clf()

    # Create figure with higher DPI and specific size
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = plt.gca()

    # Calculate the correlation matrix
    corr_matrix = data.corr(method=method)

    # Create custom colormap (red to white to green)
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)

    # Create the heatmap
    sns.heatmap(corr_matrix,
                cmap=custom_cmap,
                center=0,
                vmin=-1,
                vmax=1,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 13, 'weight': 'bold', 'family': 'Arial'},
                cbar_kws={'label': 'Correlación', 'shrink': 0.8},
                square=True,
                ax=ax)

    # Customize the plot
    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker', fontsize=15, family='Arial', weight='bold')
    ax.set_ylabel('Ticker', fontsize=15, family='Arial', weight='bold')

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Get the y-axis tick labels
    yticklabels = ax.get_yticklabels()

    # Align left labels to the right
    for label in yticklabels:
        label.set_horizontalalignment('right')

    # Align right labels to the left
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)

    # Get the y-axis tick labels again after enabling right labels
    yticklabels_right = ax.get_yticklabels()

    # Align right labels to the left
    for label in yticklabels_right:
        if label.get_position()[0] > 0: # Check if the label is on the right side
            label.set_horizontalalignment('left')

    # Customize tick labels size
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True, left=True, labelleft=True)

    # Add watermark
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    # Adjust layout
    plt.tight_layout()

    return fig

def main():
    # Add data source selection
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data']
    selected_source = st.sidebar.selectbox('Seleccionar Fuente de Datos', data_sources)

    # Add correlation method selection
    correlation_methods = ['pearson', 'spearman', 'kendall']
    selected_method = st.sidebar.selectbox('Seleccionar Método de Correlación', correlation_methods,
                                           help="""
                                           **Pearson:**
                                           *   **¿Qué mide?** Imagina que tienes dos acciones. Pearson te dice si, en general, cuando una sube, la otra también sube (o baja). Es como ver si dos amigos suelen ir a los mismos lugares.
                                           *   **¿Cuándo usarlo?** Es útil cuando crees que las acciones se mueven juntas de forma muy similar, como si fueran en la misma dirección.
                                           *   **Pros:** Es fácil de entender y muy común.
                                           *   **Contras:** Si hay movimientos muy raros en los precios (como un día que una acción sube muchísimo sin razón), puede darte una idea equivocada. También, si la relación no es tan directa, no funciona tan bien.

                                           **Spearman:**
                                           *   **¿Qué mide?** En lugar de ver si las acciones suben o bajan juntas, Spearman mira si el *orden* de los precios es similar. Si una acción está en el puesto 1 en un día y la otra también está alta, Spearman lo nota. Es como ver si dos amigos suelen estar en los mismos puestos en una carrera.
                                           *   **¿Cuándo usarlo?** Es bueno cuando no estás seguro de si la relación es directa. Si una acción siempre sube cuando la otra sube, aunque no sea en la misma cantidad, Spearman lo detecta.
                                           *   **Pros:** No le importan tanto los valores raros y funciona bien aunque la relación no sea tan directa.
                                           *   **Contras:** Puede ser menos preciso que Pearson si la relación es realmente muy directa.

                                           **Kendall:**
                                           *   **¿Qué mide?** Similar a Spearman, Kendall mira el *orden* de los precios, pero de una forma un poco diferente. Se fija en si los pares de precios se mueven en la misma dirección. Es como ver si dos amigos suelen ir juntos hacia adelante o hacia atrás en una fila.
                                           *   **¿Cuándo usarlo?** Es útil cuando tienes muchos datos y quieres una medida de correlación que no se vea tan afectada por movimientos raros.
                                           *   **Pros:** Es menos sensible a valores raros y funciona bien con datos que tienen un orden claro.
                                           *   **Contras:** Puede ser más lento de calcular y menos preciso que Pearson si la relación es muy directa.
                                           """)

    # Add moving average periods selection
    ma_periods = st.sidebar.number_input("Periodos de Media Móvil", min_value=1, value=1, step=1)

    with st.sidebar:
        tickers = st.text_input(
            "Ingrese hasta 13 Tickers o Relaciones (separados por comas, ej: AAPL, MSFT, AAPL/MSFT)",
            value="AAPL, MSFT, GOOG, TSLA, AMZN, NVDA, META, UNH, JPM, V, XOM"
        )
        start_date = st.date_input("Fecha de Inicio", value=pd.to_datetime("2023-01-01"),min_value=pd.to_datetime("1980-01-01"))
        end_date = st.date_input("Fecha de Fin", value=pd.to_datetime("2024-12-31"))
        confirm_data = st.button("Confirmar Datos")

    if confirm_data:
        try:
            with st.spinner('Obteniendo y procesando datos...'):
                ticker_list = [ticker.strip() for ticker in tickers.split(",")]
                if len(ticker_list) > 13:
                    st.error("Por favor, ingrese un máximo de 13 tickers o relaciones.")
                else:
                    correlation_data = prepare_correlation_data(ticker_list, start_date, end_date, selected_source, ma_periods)
                    if not correlation_data.empty:
                        fig = plot_correlation_matrix(correlation_data, f'Matriz de Correlación desde {start_date.strftime("%Y-%m-%d")} hasta {end_date.strftime("%Y-%m-%d")}', method=selected_method)
                        st.pyplot(fig, dpi=300)
                    else:
                        st.error("No hay datos disponibles para los tickers y el rango de fechas seleccionados.")

        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")
            st.info("Por favor, verifique si los símbolos de los tickers son válidos y si el rango de fechas es apropiado.")

if __name__ == "__main__":
    main()
