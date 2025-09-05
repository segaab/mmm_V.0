# Streamlit UI components
def render_tradingview_chart(symbol, chart_data, container_id, chart_type="Candlestick", height=400):
    # Apply appropriate data transformation based on chart type
    if chart_type == "Heikin-Ashi":
        chart_data = calculate_heikin_ashi(chart_data)
    
    # Convert chart data to JSON
    chart_data_json = json.dumps(chart_data)
    
    # Generate custom chart with Lightweight Charts
    js_code = f"""
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <div id="{container_id}" style="width: 100%; height: {height}px;"></div>
    <script>
        // Function to create and update chart
        function createChart() {{
            const chartData = {chart_data_json};
            
            const chart = LightweightCharts.createChart(
                document.getElementById('{container_id}'), 
                {{
                    layout: {{
                        background: {{ color: '#222' }},
                        textColor: '#DDD',
                    }},
                    grid: {{
                        vertLines: {{ color: '#444' }},
                        horzLines: {{ color: '#444' }},
                    }},
                    crosshair: {{
                        mode: LightweightCharts.CrosshairMode.Normal,
                    }},
                    rightPriceScale: {{
                        borderColor: '#666',
                    }},
                    timeScale: {{
                        borderColor: '#666',
                    }},
                    width: document.getElementById('{container_id}').clientWidth,
                    height: {height},
                }}
            );

            // Create series based on chart type
            let series;
            const chartType = "{chart_type}";
            
            if (chartType === "Line") {{
                series = chart.addLineSeries({{
                    color: '#2962FF',
                    lineWidth: 2,
                    crosshairMarkerVisible: true,
                }});
                
                // Format data for line chart
                const lineData = chartData.map(item => ({{
                    time: item.time,
                    value: item.close
                }}));
                
                series.setData(lineData);
            }} else {{
                // For both Candlestick and Heikin-Ashi
                series = chart.addCandlestickSeries({{
                    upColor: '#26a69a', 
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a', 
                    wickDownColor: '#ef5350'
                }});
                
                series.setData(chartData);
            }}
            
            // Add volume series
            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{
                    type: 'volume',
                }},
                priceScaleId: '',
                scaleMargins: {{
                    top: 0.8,
                    bottom: 0,
                }},
            }});
            
            // Format volume data
            const volumeData = chartData.map(item => ({{
                time: item.time,
                value: item.volume,
                color: item.close > item.open ? '#26a69a' : '#ef5350',
            }}));
            
            volumeSeries.setData(volumeData);
            
            // Create tooltip
            const toolTipWidth = 100;
            const toolTipHeight = 100;
            const toolTipMargin = 15;
            
            // Create and style the tooltip element
            const toolTip = document.createElement('div');
            toolTip.style = `width: ${{toolTipWidth}}px; height: ${{toolTipHeight}}px; position: absolute; display: none; padding: 8px; box-sizing: border-box; font-size: 12px; color: #fff; background-color: rgba(0, 0, 0, 0.7); border-radius: 4px; z-index: 1000; top: 12px; left: 12px; pointer-events: none;`;
            document.getElementById('{container_id}').appendChild(toolTip);
            
            // Chart mouse hover event
            chart.subscribeCrosshairMove((param) => {{
                if (!param.point || !param.time || param.point.x < 0 || param.point.x > chart.clientWidth || param.point.y < 0 || param.point.y > chart.clientHeight) {{
                    toolTip.style.display = 'none';
                    return;
                }}
                
                const dateStr = new Date(param.time * 1000).toLocaleDateString();
                let dataPoint;
                
                if (chartType === "Line") {{
                    dataPoint = param.seriesData.get(series);
                    if (dataPoint) {{
                        // Find the original OHLC data for this time point
                        const originalPoint = chartData.find(d => d.time === param.time);
                        if (originalPoint) {{
                            dataPoint = {{
                                value: dataPoint.value,
                                open: originalPoint.open,
                                high: originalPoint.high,
                                low: originalPoint.low,
                                close: originalPoint.close
                            }};
                        }}
                    }}
                }} else {{
                    dataPoint = param.seriesData.get(series);
                }}
                
                toolTip.style.display = 'block';
                const volumePoint = param.seriesData.get(volumeSeries);
                
                let content = `<div style="font-size: 12px; margin: 4px 0px;">${{dateStr}}</div>`;
                
                if (dataPoint) {{
                    if (chartType === "Line") {{
                        content += `
                            <div>O: ${{dataPoint.open?.toFixed(2) || 'N/A'}}</div>
                            <div>H: ${{dataPoint.high?.toFixed(2) || 'N/A'}}</div>
                            <div>L: ${{dataPoint.low?.toFixed(2) || 'N/A'}}</div>
                            <div>C: ${{dataPoint.close?.toFixed(2) || dataPoint.value?.toFixed(2) || 'N/A'}}</div>
                        `;
                    }} else {{
                        content += `
                            <div>O: ${{dataPoint.open?.toFixed(2) || 'N/A'}}</div>
                            <div>H: ${{dataPoint.high?.toFixed(2) || 'N/A'}}</div>
                            <div>L: ${{dataPoint.low?.toFixed(2) || 'N/A'}}</div>
                            <div>C: ${{dataPoint.close?.toFixed(2) || 'N/A'}}</div>
                        `;
                    }}
                    
                    if (volumePoint) {{
                        content += `<div>V: ${{volumePoint.value}}</div>`;
                    }}
                }}
                
                toolTip.innerHTML = content;
                
                // Position tooltip
                let left = param.point.x - 50;
                if (left < toolTipMargin) {{
                    left = toolTipMargin;
                }}
                if (left + toolTipWidth + toolTipMargin > chart.clientWidth) {{
                    left = chart.clientWidth - toolTipWidth - toolTipMargin;
                }}
                
                let top = toolTipMargin;
                if (param.point.y - toolTipHeight - toolTipMargin > 0) {{
                    top = param.point.y - toolTipHeight - toolTipMargin;
                }} else {{
                    top = param.point.y + toolTipMargin;
                }}
                
                toolTip.style.left = left + 'px';
                toolTip.style.top = top + 'px';
            }});
            
            // Handle resize
            function handleResize() {{
                chart.applyOptions({{ 
                    width: document.getElementById('{container_id}').clientWidth 
                }});
            }}
            
            window.addEventListener('resize', handleResize);
            
            // Add chart title
            const chartTitle = document.createElement('div');
            chartTitle.style = `position: absolute; top: 5px; left: 10px; font-size: 16px; font-weight: bold; color: #DDD; z-index: 100;`;
            chartTitle.textContent = "{symbol} ({chart_type})";
            document.getElementById('{container_id}').appendChild(chartTitle);
        }}
        
        // Call function to create chart when DOM is loaded
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', createChart);
        }} else {{
            createChart();
        }}
    </script>
    """
    
    return js_code

def build_dashboard():
    st.title("Market Monitor Dashboard")
    
    # Initialize session state for selections and last update time
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now() - timedelta(minutes=10)  # Force update on first run
    
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = list(MARKETS.keys())[0]
    
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = MARKETS[st.session_state.selected_category][0]
    
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = "1d"
    
    if 'selected_chart_type' not in st.session_state:
        st.session_state.selected_chart_type = "Candlestick"
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Market and symbol selector
        col1, col2 = st.columns(2)
        
        with col1:
            selected_category = st.selectbox(
                "Market Category",
                list(MARKETS.keys()),
                index=list(MARKETS.keys()).index(st.session_state.selected_category),
                key="category_selector"
            )
        
        # Update selected category in session state
        if selected_category != st.session_state.selected_category:
            st.session_state.selected_category = selected_category
            # Reset symbol to first in category when category changes
            st.session_state.selected_symbol = MARKETS[selected_category][0]
        
        with col2:
            available_symbols = MARKETS[selected_category]
            selected_symbol = st.selectbox(
                "Symbol",
                available_symbols,
                index=available_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in available_symbols else 0,
                key="symbol_selector"
            )
        
        # Update selected symbol in session state
        if selected_symbol != st.session_state.selected_symbol:
            st.session_state.selected_symbol = selected_symbol
        
        # Chart settings
        st.subheader("Chart Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            selected_timeframe = st.selectbox(
                "Timeframe",
                list(TIMEFRAMES.keys()),
                index=list(TIMEFRAMES.keys()).index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in TIMEFRAMES else 2,
                key="timeframe_selector"
            )
        
        # Update selected timeframe in session state
        if selected_timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = selected_timeframe
        
        with col4:
            selected_chart_type = st.selectbox(
                "Chart Type",
                CHART_TYPES,
                index=CHART_TYPES.index(st.session_state.selected_chart_type) if st.session_state.selected_chart_type in CHART_TYPES else 0,
                key="chart_type_selector"
            )
        
        # Update selected chart type in session state
        if selected_chart_type != st.session_state.selected_chart_type:
            st.session_state.selected_chart_type = selected_chart_type
        
        # History range
        days_lookback = st.slider(
            "Days of History", 
            min_value=7, 
            max_value=90, 
            value=30,
            key="days_lookback_slider"
        )
        
        # Data refresh settings
        st.header("Data Refresh Settings")
        use_real_api = st.checkbox("Use Real API (if API Key provided)", value=True)
        refresh_minutes = st.number_input("Update Frequency (minutes)", min_value=1, max_value=360, value=60)
        refresh_seconds = refresh_minutes * 60
        
        # Manual refresh button
        if st.button("Refresh Data Now"):
            update_live_prices([selected_symbol], use_real_api)
            st.session_state.last_update_time = datetime.now()
            st.success(f"Data updated at {st.session_state.last_update_time.strftime('%H:%M:%S')}")
        
        # Alert creation
        st.header("Create Alert")
        
        # Get current price or estimate
        current_data = cache.get_live(selected_symbol)
        current_price = current_data["price"] if current_data else 100
        
        alert_price = st.number_input("Alert Price", value=current_price, step=0.01)
        alert_condition = st.radio("Condition", ["above", "below"])
        alert_message = st.text_input("Alert Message", value=f"{selected_symbol} {alert_condition} {alert_price}")
        
        if st.button("Add Alert"):
            alert_system.add_alert(selected_symbol, alert_condition, alert_price, alert_message)
            st.success(f"Alert added for {selected_symbol}")
        
        # Display active alerts
        st.header("Active Alerts")
        active_alerts = alert_system.get_active_alerts(selected_symbol)
        for i, alert in enumerate(active_alerts):
            st.write(f"{i+1}. {alert['condition'].title()} {alert['price']}: {alert['message']}")
        
        # Display time since last update
        time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds() / 60
        st.write(f"Last data update: {time_since_update:.1f} minutes ago")
    
    # Check if it's time to update data
    time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
    if time_since_update > refresh_seconds:
        update_live_prices([selected_symbol], use_real_api)
        st.session_state.last_update_time = datetime.now()
    
    # Main dashboard area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main chart for selected symbol
        timeframe_api = TIMEFRAMES[selected_timeframe]
        
        # Adjust days for timeframe
        adjusted_days = days_lookback
        if selected_timeframe == "1h":
            adjusted_days = min(days_lookback, 7)  # Limit 1h to 7 days for realistic data amount
        elif selected_timeframe == "4h":
            adjusted_days = min(days_lookback, 30)  # Limit 4h to 30 days
        
        # Main chart for selected symbol
        st.header(f"{selected_symbol} Chart")
        chart_container_id = f"chart_{selected_symbol}_{timeframe_api}_{selected_chart_type}"
        chart_data = prepare_chart_data(selected_symbol, timeframe_api, adjusted_days)
        
        # Create HTML component for the chart
        chart_html = render_tradingview_chart(
            selected_symbol, 
            chart_data, 
            chart_container_id, 
            chart_type=selected_chart_type,
            height=500
        )
        st.components.v1.html(chart_html, height=520)
        
        # Technical analysis indicators
        with st.expander("Technical Analysis"):
            ta_col1, ta_col2, ta_col3 = st.columns(3)
            
            with ta_col1:
                st.metric("RSI", f"{random.randint(30, 70)}", f"{random.choice(['+', '-'])}{random.randint(1, 5)}")
            
            with ta_col2:
                st.metric("MACD", f"{random.choice(['+', '-'])}{random.randint(1, 100)/100}", 
                          f"{random.choice(['+', '-'])}{random.randint(1, 20)/100}")
            
            with ta_col3:
                st.metric("Bollinger Bands", f"{random.randint(0, 100)}% B", 
                          f"{random.choice(['+', '-'])}{random.randint(1, 10)}%")
    
    with col2:
        # Market overview table
        st.header("Market Overview")
        
        # Get current data for all symbols in the selected category
        live_data = {}
        for symbol in MARKETS[selected_category]:
            # Get live data from cache or generate placeholder
            symbol_data = cache.get_live(symbol)
            if symbol_data:
                price = symbol_data["price"]
                timestamp = symbol_data["timestamp"]
            else:
                # Generate placeholder data
                base = 100
                if symbol.startswith("EUR"):
                    base = 1.1
                elif symbol.startswith("GBP"):
                    base = 1.3
                elif symbol.startswith("USD"):
                    base = 0.9
                elif symbol.startswith("XAU"):
                    base = 2000
                
                price = base * (1 + (random.random() - 0.5) * 0.01)
                timestamp = int(time.time() * 1000)
            
            # Calculate dummy change (would be real in production)
            change = (random.random() - 0.5) * 0.5  # -0.25% to +0.25%
            
            live_data[symbol] = {
                "price": price,
                "change": change,
                "timestamp": timestamp
            }
            
            # Check for any triggered alerts
            if symbol_data:
                triggered_alerts = alert_system.check_alerts(symbol, price)
                for alert in triggered_alerts:
                    st.warning(f"ALERT: {alert}")
        
        # Create dataframe for the table
        market_data = []
        for symbol, data in live_data.items():
            market_data.append({
                "Symbol": symbol,
                "Price": round(data["price"], 4),
                "Change %": round(data["change"], 2),
                "Updated": datetime.fromtimestamp(data["timestamp"]/1000).strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(market_data)
        
        # Style the dataframe
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = 'red' if val < 0 else 'green'
                return f'color: {color}'
            return ''
        
        styled_df = df.style.applymap(color_negative_red, subset=['Change %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Recent alerts
        st.header("Recent Alerts")
        all_alerts = []
        for symbol, alerts in alert_system.alerts.items():
            for alert in alerts:
                all_alerts.append({
                    "Symbol": symbol,
                    "Condition": alert["condition"],
                    "Price": alert["price"],
                    "Message": alert["message"],
                    "Created": alert["created_at"].strftime('%H:%M:%S')
                })
        
        if all_alerts:
            alerts_df = pd.DataFrame(all_alerts)
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No active alerts")
        
        # Market calendar/news
        st.header("Market Calendar")
        today = datetime.now().date()
        
        # Dummy market events
        events = [
            {"time": "08:30", "event": "US Non-Farm Payrolls", "impact": "High"},
            {"time": "10:00", "event": "ECB Interest Rate Decision", "impact": "High"},
           ]
        
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True)
    
        # Auto-refresh based on refresh_seconds
        # Note: This will refresh the entire Streamlit app, not just the data
        if refresh_seconds > 0:
           st.empty()
           time.sleep(min(refresh_seconds, 10))  # Cap at 10 seconds to avoid blocking UI
           st.experimental_rerun()

    # Run the dashboard
if __name__ == "__main__":
    build_dashboard()
