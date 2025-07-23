from AI_Stock_Trading import AlpacaPaperSocket, PMModelDevelopment, PortfolioManagementSystem


print("Training AI model...")
model_dev = PMModelDevelopment()
model_dev.train_model()
print("Model training complete!")

# 3. Start automated trading
print("Starting trading system...")
trading_system = PortfolioManagementSystem(
    api=AlpacaPaperSocket(),
    symbol="AAPL",
    time_frame="1D",
    system_id="PM_System",
    system_label="Portfolio Management System",
)
print("System is now running!")