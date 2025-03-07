# data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class AdvancedTransactionSimulator:
    def __init__(self, num_accounts=1000, num_transactions=10000, fraud_ratio=0.05):
        """
        Generate synthetic transaction data for fraud detection
        
        Args:
            num_accounts (int): Number of unique accounts
            num_transactions (int): Total number of transactions
            fraud_ratio (float): Ratio of fraudulent transactions
        """
        self.num_accounts = num_accounts
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio
        self.faker = Faker()
    
    def generate_accounts(self):
        """Generate synthetic account data"""
        accounts = []
        for i in range(self.num_accounts):
            accounts.append({
                'account_id': i,
                'age': random.randint(18, 80),  # Add 'age' column
                'income_level': random.choice(['low', 'medium', 'high'])  # Add 'income_level' column
            })
        return pd.DataFrame(accounts)
    
    def generate_transactions(self, accounts):
        """Generate synthetic transaction data"""
        transactions = []
        for i in range(self.num_transactions):
            sender = random.choice(accounts['account_id'])
            recipient = random.choice(accounts['account_id'])
            amount = random.randint(1, 10000)
            timestamp = self.faker.date_time_between(start_date='-1y', end_date='now')
            is_fraud = 1 if random.random() < self.fraud_ratio else 0
            
            transactions.append({
                'sender_id': sender,
                'recipient_id': recipient,
                'amount': amount,
                'timestamp': timestamp,
                'is_fraud': is_fraud
            })
        return pd.DataFrame(transactions)
    
    def save_dataset(self, file_path):
        """Generate and save the synthetic dataset"""
        accounts = self.generate_accounts()
        transactions = self.generate_transactions(accounts)
        
        # Merge account data with transactions
        transactions = transactions.merge(accounts, left_on='sender_id', right_on='account_id', suffixes=('', '_sender'))
        transactions = transactions.merge(accounts, left_on='recipient_id', right_on='account_id', suffixes=('', '_recipient'))
        
        # Save to CSV
        transactions.to_csv(file_path, index=False)

if __name__ == "__main__":
    simulator = AdvancedTransactionSimulator(num_accounts=5000, num_transactions=100000, fraud_ratio=0.07)
    simulator.save_dataset('data/synthetic_fraud_dataset.csv')