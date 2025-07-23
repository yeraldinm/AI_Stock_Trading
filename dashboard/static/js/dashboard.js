/**
 * Advanced Trading Dashboard - Client-side JavaScript
 */

class TradingDashboard {
    constructor() {
        this.socket = null;
        this.priceChart = null;
        this.marketData = {};
        this.portfolioData = {};
        this.riskMetrics = {};
        this.strategyPerformance = {};
        this.previousPrices = {};
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.initializeChart();
        this.setupEventListeners();
        this.updateClock();
        this.loadStrategies();
        
        // Update clock every second
        setInterval(() => this.updateClock(), 1000);
    }
    
    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('initial_data', (data) => {
            console.log('Received initial data:', data);
            this.handleInitialData(data);
        });
        
        this.socket.on('market_data_update', (data) => {
            this.handleMarketDataUpdate(data);
        });
        
        this.socket.on('portfolio_update', (data) => {
            this.handlePortfolioUpdate(data);
        });
        
        this.socket.on('new_order', (data) => {
            this.handleNewOrder(data);
        });
        
        this.socket.on('new_fill', (data) => {
            this.handleNewFill(data);
        });
        
        this.socket.on('risk_metrics_update', (data) => {
            this.handleRiskMetricsUpdate(data);
        });
        
        this.socket.on('strategy_performance_update', (data) => {
            this.handleStrategyPerformanceUpdate(data);
        });
        
        this.socket.on('strategy_status_update', (data) => {
            this.handleStrategyStatusUpdate(data);
        });
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.className = 'badge bg-success';
            statusElement.innerHTML = '<i class="bi bi-wifi"></i> Connected';
        } else {
            statusElement.className = 'badge bg-danger';
            statusElement.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
        }
    }
    
    updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: false,
            timeZone: 'America/New_York'
        });
        document.getElementById('current-time').textContent = `${timeString} EST`;
    }
    
    handleInitialData(data) {
        if (data.market_data) {
            this.marketData = data.market_data;
            this.updateMarketDataTable();
        }
        
        if (data.portfolio) {
            this.portfolioData = data.portfolio;
            this.updatePortfolioMetrics();
            this.updatePositionsTable();
        }
        
        if (data.orders) {
            this.updateOrdersTable(data.orders);
        }
        
        if (data.risk_metrics) {
            this.riskMetrics = data.risk_metrics;
            this.updateRiskMetrics();
        }
        
        if (data.strategy_performance) {
            this.strategyPerformance = data.strategy_performance;
            this.updateStrategyPerformance();
        }
    }
    
    handleMarketDataUpdate(data) {
        const { symbol, data: marketData } = data;
        
        // Store previous price for animation
        if (this.marketData[symbol]) {
            this.previousPrices[symbol] = this.marketData[symbol].price;
        }
        
        this.marketData[symbol] = marketData;
        this.updateMarketDataTable();
        
        // Update chart if this symbol is currently selected
        const selectedSymbol = document.getElementById('chart-symbol').value;
        if (symbol === selectedSymbol) {
            this.updateChart(symbol);
        }
    }
    
    handlePortfolioUpdate(data) {
        this.portfolioData = data;
        this.updatePortfolioMetrics();
        this.updatePositionsTable();
    }
    
    handleNewOrder(data) {
        this.addOrderToTable(data);
        this.showNotification('New Order', `${data.side} ${data.quantity} ${data.symbol} at ${data.order_type}`, 'info');
    }
    
    handleNewFill(data) {
        this.showNotification('Order Filled', `${data.side} ${data.quantity} ${data.symbol} at $${data.price}`, 'success');
    }
    
    handleRiskMetricsUpdate(data) {
        this.riskMetrics = data;
        this.updateRiskMetrics();
    }
    
    handleStrategyPerformanceUpdate(data) {
        const { strategy, data: performanceData } = data;
        this.strategyPerformance[strategy] = performanceData;
        this.updateStrategyPerformance();
    }
    
    updateMarketDataTable() {
        const tbody = document.getElementById('market-data-table');
        tbody.innerHTML = '';
        
        Object.entries(this.marketData).forEach(([symbol, data]) => {
            const row = document.createElement('tr');
            const previousPrice = this.previousPrices[symbol];
            const currentPrice = data.price;
            
            let priceClass = '';
            if (previousPrice !== undefined) {
                if (currentPrice > previousPrice) {
                    priceClass = 'price-up';
                } else if (currentPrice < previousPrice) {
                    priceClass = 'price-down';
                }
            }
            
            const changeClass = data.change >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = data.change >= 0 ? 'bi-arrow-up' : 'bi-arrow-down';
            
            row.innerHTML = `
                <td><strong>${symbol}</strong></td>
                <td class="${priceClass}">$${data.price.toFixed(2)}</td>
                <td class="${changeClass}">
                    <i class="bi ${changeIcon}"></i> $${Math.abs(data.change).toFixed(2)}
                </td>
                <td class="${changeClass}">${data.change_percent.toFixed(2)}%</td>
                <td>${data.volume.toLocaleString()}</td>
                <td>$${data.bid.toFixed(2)}</td>
                <td>$${data.ask.toFixed(2)}</td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    updatePortfolioMetrics() {
        if (!this.portfolioData) return;
        
        document.getElementById('total-value').textContent = `$${this.portfolioData.total_value?.toLocaleString() || '0.00'}`;
        document.getElementById('cash-balance').textContent = `$${this.portfolioData.cash?.toLocaleString() || '0.00'}`;
        
        const pnlToday = this.portfolioData.pnl_today || 0;
        const pnlTodayElement = document.getElementById('pnl-today');
        pnlTodayElement.textContent = `$${pnlToday.toLocaleString()}`;
        pnlTodayElement.className = pnlToday >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
        
        const pnlTotal = this.portfolioData.pnl_total || 0;
        const pnlTotalElement = document.getElementById('pnl-total');
        pnlTotalElement.textContent = `$${pnlTotal.toLocaleString()}`;
        pnlTotalElement.className = pnlTotal >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
    }
    
    updatePositionsTable() {
        const tbody = document.getElementById('positions-table');
        tbody.innerHTML = '';
        
        if (!this.portfolioData.positions) return;
        
        this.portfolioData.positions.forEach(position => {
            const row = document.createElement('tr');
            const pnlClass = position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
            const quantityClass = position.quantity >= 0 ? 'text-success' : 'text-danger';
            
            row.innerHTML = `
                <td><strong>${position.symbol}</strong></td>
                <td class="${quantityClass}">${position.quantity}</td>
                <td>$${position.avg_price.toFixed(2)}</td>
                <td>$${position.market_value.toLocaleString()}</td>
                <td class="${pnlClass}">$${position.unrealized_pnl.toFixed(2)}</td>
                <td>
                    <button class="btn btn-sm btn-outline-danger" onclick="dashboard.closePosition('${position.symbol}')">
                        Close
                    </button>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    updateOrdersTable(orders) {
        const tbody = document.getElementById('orders-table');
        tbody.innerHTML = '';
        
        // Show only the most recent 20 orders
        const recentOrders = orders.slice(-20).reverse();
        
        recentOrders.forEach(order => {
            this.addOrderToTable(order);
        });
    }
    
    addOrderToTable(order) {
        const tbody = document.getElementById('orders-table');
        const row = document.createElement('tr');
        
        let statusClass = 'bg-secondary';
        switch (order.status) {
            case 'FILLED':
                statusClass = 'bg-success';
                break;
            case 'CANCELLED':
                statusClass = 'bg-danger';
                break;
            case 'PENDING':
                statusClass = 'bg-warning';
                break;
        }
        
        const sideClass = order.side === 'BUY' ? 'text-success' : 'text-danger';
        
        row.innerHTML = `
            <td><strong>${order.symbol}</strong></td>
            <td class="${sideClass}">${order.side}</td>
            <td>${order.quantity}</td>
            <td><span class="badge ${statusClass} status-badge">${order.status}</span></td>
        `;
        
        // Insert at the beginning
        tbody.insertBefore(row, tbody.firstChild);
        
        // Keep only 20 rows
        while (tbody.children.length > 20) {
            tbody.removeChild(tbody.lastChild);
        }
    }
    
    updateRiskMetrics() {
        if (!this.riskMetrics) return;
        
        document.getElementById('var-95').textContent = `$${this.riskMetrics.var_95?.toLocaleString() || '0.00'}`;
        document.getElementById('max-drawdown').textContent = `${(this.riskMetrics.max_drawdown * 100).toFixed(2)}%`;
        document.getElementById('sharpe-ratio').textContent = this.riskMetrics.sharpe_ratio?.toFixed(2) || '0.00';
        document.getElementById('volatility').textContent = `${(this.riskMetrics.volatility * 100).toFixed(2)}%`;
        document.getElementById('beta').textContent = this.riskMetrics.beta?.toFixed(2) || '0.00';
        document.getElementById('exposure').textContent = `${(this.riskMetrics.exposure * 100).toFixed(2)}%`;
    }
    
    updateStrategyPerformance() {
        const container = document.getElementById('strategy-performance');
        container.innerHTML = '';
        
        Object.entries(this.strategyPerformance).forEach(([strategy, data]) => {
            const pnlClass = data.pnl >= 0 ? 'text-success' : 'text-danger';
            const indicatorClass = data.pnl >= 0 ? 'indicator-green' : 'indicator-red';
            
            const strategyDiv = document.createElement('div');
            strategyDiv.className = 'mb-3 p-2 border border-secondary rounded';
            strategyDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">
                        <span class="performance-indicator ${indicatorClass}"></span>
                        ${strategy}
                    </h6>
                    <small class="text-secondary">${data.trades_today} trades</small>
                </div>
                <div class="row g-2">
                    <div class="col-6">
                        <small class="text-secondary">P&L</small>
                        <div class="fw-bold ${pnlClass}">$${data.pnl.toFixed(2)}</div>
                    </div>
                    <div class="col-6">
                        <small class="text-secondary">Win Rate</small>
                        <div class="fw-bold">${(data.win_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="col-6">
                        <small class="text-secondary">Sharpe</small>
                        <div class="fw-bold">${data.sharpe_ratio.toFixed(2)}</div>
                    </div>
                    <div class="col-6">
                        <small class="text-secondary">Avg Trade</small>
                        <div class="fw-bold">$${data.avg_trade.toFixed(2)}</div>
                    </div>
                </div>
            `;
            
            container.appendChild(strategyDiv);
        });
    }
    
    initializeChart() {
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        this.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#1f6feb',
                    backgroundColor: 'rgba(31, 111, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: '#30363d'
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    },
                    y: {
                        grid: {
                            color: '#30363d'
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    }
                }
            }
        });
        
        this.loadHistoricalData('AAPL');
    }
    
    async loadHistoricalData(symbol) {
        try {
            const response = await fetch(`/api/historical-data/${symbol}`);
            const data = await response.json();
            
            const labels = data.map(item => new Date(item.timestamp).toLocaleTimeString());
            const prices = data.map(item => item.close);
            
            this.priceChart.data.labels = labels;
            this.priceChart.data.datasets[0].data = prices;
            this.priceChart.data.datasets[0].label = `${symbol} Price`;
            this.priceChart.update();
            
        } catch (error) {
            console.error('Error loading historical data:', error);
        }
    }
    
    updateChart(symbol) {
        // In a real implementation, you would update the chart with real-time data
        // For now, we'll just add the latest price point
        if (this.marketData[symbol] && this.priceChart) {
            const currentTime = new Date().toLocaleTimeString();
            const currentPrice = this.marketData[symbol].price;
            
            // Add new data point
            this.priceChart.data.labels.push(currentTime);
            this.priceChart.data.datasets[0].data.push(currentPrice);
            
            // Keep only last 50 points
            if (this.priceChart.data.labels.length > 50) {
                this.priceChart.data.labels.shift();
                this.priceChart.data.datasets[0].data.shift();
            }
            
            this.priceChart.update('none');
        }
    }
    
    setupEventListeners() {
        // Chart symbol selector
        document.getElementById('chart-symbol').addEventListener('change', (e) => {
            this.loadHistoricalData(e.target.value);
        });
        
        // Strategy management buttons
        document.getElementById('create-strategy-btn').addEventListener('click', () => {
            const modal = new bootstrap.Modal(document.getElementById('createStrategyModal'));
            modal.show();
        });
        
        document.getElementById('refresh-strategies-btn').addEventListener('click', () => {
            this.loadStrategies();
        });
        
        document.getElementById('create-strategy-submit').addEventListener('click', () => {
            this.createStrategy();
        });
        
        // Trade form
        document.getElementById('trade-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.placeOrder();
        });
        
        // Order type change
        document.getElementById('trade-type').addEventListener('change', (e) => {
            const priceField = document.getElementById('price-field');
            if (e.target.value === 'LIMIT' || e.target.value === 'STOP') {
                priceField.style.display = 'block';
                document.getElementById('trade-price').required = true;
            } else {
                priceField.style.display = 'none';
                document.getElementById('trade-price').required = false;
            }
        });
        
        // Market data view buttons
        document.querySelectorAll('[data-view]').forEach(button => {
            button.addEventListener('click', (e) => {
                // Update active button
                document.querySelectorAll('[data-view]').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                
                // Filter market data based on view
                this.filterMarketData(e.target.dataset.view);
            });
        });
    }
    
    async placeOrder() {
        const formData = {
            symbol: document.getElementById('trade-symbol').value,
            side: document.getElementById('trade-side').value,
            quantity: parseInt(document.getElementById('trade-quantity').value),
            order_type: document.getElementById('trade-type').value,
            price: parseFloat(document.getElementById('trade-price').value) || 0
        };
        
        try {
            const response = await fetch('/api/place-order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Order Placed', `${formData.side} order for ${formData.quantity} ${formData.symbol} submitted`, 'success');
                document.getElementById('trade-form').reset();
                document.getElementById('price-field').style.display = 'none';
            } else {
                this.showNotification('Order Failed', result.error, 'danger');
            }
            
        } catch (error) {
            console.error('Error placing order:', error);
            this.showNotification('Order Failed', 'Network error occurred', 'danger');
        }
    }
    
    closePosition(symbol) {
        // In a real implementation, this would close the position
        this.showNotification('Position Closed', `Closing position for ${symbol}`, 'info');
    }
    
    filterMarketData(view) {
        // In a real implementation, this would filter the market data table
        // based on the selected view (watchlist, gainers, losers)
        console.log(`Filtering market data by: ${view}`);
    }
    
    showNotification(title, message, type = 'info') {
        const toast = document.getElementById('notification-toast');
        const toastHeader = toast.querySelector('.toast-header strong');
        const toastBody = toast.querySelector('.toast-body');
        
        toastHeader.textContent = title;
        toastBody.textContent = message;
        
        // Update toast style based on type
        toast.className = `toast border-${type}`;
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
    
    // Strategy Management Methods
    async loadStrategies() {
        try {
            const response = await fetch('/api/strategies');
            const strategies = await response.json();
            this.updateStrategiesTable(strategies);
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
    }
    
    updateStrategiesTable(strategies) {
        const tableBody = document.getElementById('strategies-table');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        Object.values(strategies).forEach(strategy => {
            const row = document.createElement('tr');
            
            const statusBadge = strategy.enabled ? 
                '<span class="badge bg-success">Running</span>' : 
                '<span class="badge bg-secondary">Stopped</span>';
            
            const pnl = strategy.performance?.total_pnl || 0;
            const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
            const pnlSymbol = pnl >= 0 ? '+' : '';
            
            row.innerHTML = `
                <td>
                    <strong>${strategy.name}</strong><br>
                    <small class="text-muted">${strategy.id}</small>
                </td>
                <td>${statusBadge}</td>
                <td>
                    <small>${strategy.symbols.join(', ')}</small>
                </td>
                <td class="${pnlClass}">
                    ${pnlSymbol}$${pnl.toFixed(2)}
                </td>
                <td>
                    <span class="badge bg-info">${strategy.performance?.active_positions || 0}</span>
                </td>
                <td>
                    <span class="badge bg-warning">${strategy.performance?.active_orders || 0}</span>
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        ${strategy.enabled ? 
                            `<button class="btn btn-outline-warning" onclick="dashboard.stopStrategy('${strategy.id}')">
                                <i class="bi bi-pause-fill"></i>
                            </button>` :
                            `<button class="btn btn-outline-success" onclick="dashboard.startStrategy('${strategy.id}')">
                                <i class="bi bi-play-fill"></i>
                            </button>`
                        }
                        <button class="btn btn-outline-info" onclick="dashboard.showStrategyDetails('${strategy.id}')">
                            <i class="bi bi-info-circle"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="dashboard.deleteStrategy('${strategy.id}')">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    async startStrategy(strategyId) {
        try {
            const response = await fetch(`/api/strategies/${strategyId}/start`, {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('Strategy Started', result.message, 'success');
                this.loadStrategies();
            } else {
                this.showNotification('Error', result.message, 'danger');
            }
        } catch (error) {
            console.error('Error starting strategy:', error);
            this.showNotification('Error', 'Failed to start strategy', 'danger');
        }
    }
    
    async stopStrategy(strategyId) {
        try {
            const response = await fetch(`/api/strategies/${strategyId}/stop`, {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('Strategy Stopped', result.message, 'warning');
                this.loadStrategies();
            } else {
                this.showNotification('Error', result.message, 'danger');
            }
        } catch (error) {
            console.error('Error stopping strategy:', error);
            this.showNotification('Error', 'Failed to stop strategy', 'danger');
        }
    }
    
    async deleteStrategy(strategyId) {
        if (!confirm('Are you sure you want to delete this strategy?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/strategies/${strategyId}/delete`, {
                method: 'DELETE'
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('Strategy Deleted', result.message, 'info');
                this.loadStrategies();
            } else {
                this.showNotification('Error', result.message, 'danger');
            }
        } catch (error) {
            console.error('Error deleting strategy:', error);
            this.showNotification('Error', 'Failed to delete strategy', 'danger');
        }
    }
    
    async showStrategyDetails(strategyId) {
        try {
            const response = await fetch(`/api/strategies/${strategyId}/performance`);
            const data = await response.json();
            
            // Update modal content
            this.updateStrategyModal(data);
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('strategyModal'));
            modal.show();
            
        } catch (error) {
            console.error('Error loading strategy details:', error);
            this.showNotification('Error', 'Failed to load strategy details', 'danger');
        }
    }
    
    updateStrategyModal(data) {
        const metricsDiv = document.getElementById('strategy-metrics');
        const positionsDiv = document.getElementById('strategy-positions');
        const tradesDiv = document.getElementById('strategy-trades');
        
        // Update metrics
        const metrics = data.basic_metrics || {};
        metricsDiv.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <div class="metric-item">
                        <small class="text-muted">Total P&L</small>
                        <div class="${metrics.total_pnl >= 0 ? 'text-success' : 'text-danger'}">
                            $${(metrics.total_pnl || 0).toFixed(2)}
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="metric-item">
                        <small class="text-muted">Win Rate</small>
                        <div>${((metrics.win_rate || 0) * 100).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="metric-item">
                        <small class="text-muted">Total Trades</small>
                        <div>${metrics.total_trades || 0}</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="metric-item">
                        <small class="text-muted">Signals Generated</small>
                        <div>${metrics.signals_generated || 0}</div>
                    </div>
                </div>
            </div>
        `;
        
        // Update positions
        const positions = data.positions || {};
        if (Object.keys(positions).length === 0) {
            positionsDiv.innerHTML = '<p class="text-muted">No active positions</p>';
        } else {
            let positionsHtml = '<div class="table-responsive"><table class="table table-sm table-dark"><thead><tr><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>P&L</th></tr></thead><tbody>';
            
            Object.values(positions).forEach(pos => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
                positionsHtml += `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.quantity}</td>
                        <td>$${pos.average_price.toFixed(2)}</td>
                        <td class="${pnlClass}">$${pos.unrealized_pnl.toFixed(2)}</td>
                    </tr>
                `;
            });
            
            positionsHtml += '</tbody></table></div>';
            positionsDiv.innerHTML = positionsHtml;
        }
        
        // Update trades
        const trades = data.trade_history || [];
        if (trades.length === 0) {
            tradesDiv.innerHTML = '<p class="text-muted">No recent trades</p>';
        } else {
            let tradesHtml = '<div class="table-responsive"><table class="table table-sm table-dark"><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead><tbody>';
            
            trades.slice(-10).forEach(trade => {
                const time = new Date(trade.timestamp || trade.created_at).toLocaleTimeString();
                tradesHtml += `
                    <tr>
                        <td>${time}</td>
                        <td>${trade.symbol}</td>
                        <td><span class="badge ${trade.side === 'buy' ? 'bg-success' : 'bg-danger'}">${trade.side}</span></td>
                        <td>${trade.quantity}</td>
                        <td>$${(trade.price || 0).toFixed(2)}</td>
                    </tr>
                `;
            });
            
            tradesHtml += '</tbody></table></div>';
            tradesDiv.innerHTML = tradesHtml;
        }
    }
    
    async createStrategy() {
        const form = document.getElementById('create-strategy-form');
        const formData = new FormData(form);
        
        const strategyData = {
            type: document.getElementById('strategy-type').value,
            symbols: document.getElementById('strategy-symbols').value.split(',').map(s => s.trim()),
            parameters: {
                lookback_period: parseInt(document.getElementById('lookback-period').value),
                deviation_threshold: parseFloat(document.getElementById('deviation-threshold').value),
                max_position_size: parseFloat(document.getElementById('max-position-size').value),
                stop_loss: parseFloat(document.getElementById('stop-loss').value) / 100
            }
        };
        
        try {
            const response = await fetch('/api/strategies/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(strategyData)
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('Strategy Created', result.message, 'success');
                
                // Close modal and refresh strategies
                const modal = bootstrap.Modal.getInstance(document.getElementById('createStrategyModal'));
                modal.hide();
                form.reset();
                this.loadStrategies();
            } else {
                this.showNotification('Error', result.error || result.message, 'danger');
            }
        } catch (error) {
            console.error('Error creating strategy:', error);
            this.showNotification('Error', 'Failed to create strategy', 'danger');
        }
    }
    
    handleStrategyStatusUpdate(data) {
        // Refresh strategies table when status updates are received
        this.loadStrategies();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});

// Handle page visibility change to reconnect WebSocket if needed
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && (!window.dashboard.socket || !window.dashboard.socket.connected)) {
        console.log('Page became visible, reconnecting WebSocket...');
        window.dashboard.connectWebSocket();
    }
});