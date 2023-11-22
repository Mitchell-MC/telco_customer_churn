# telco_customer_churn

Project Overview

Problem Area:

    The goal is to identify customers at high risk of churn, utilizing their subscription services, demographic information, and tenure with the company.

Proposed Data Science Solution:

    Develop a predictive model to identify customers likely to churn, enabling targeted retention strategies.

Impact of the Solution:

    Improve customer retention, reduce churn rates, and enhance long-term profitability.

Dataset Description

    Size: 7,043 customers (after cleaning).
    Features: Includes demographic details (Gender, Age, etc.), account information (Tenure, Monthly Charges, etc.), and service usage (Internet Service, Online Security, etc.).
    Target Variable: Churn Label (Yes/No).

Data Dictionary:

    CustomerID
        Description: Unique identifier for each customer.
        Type: String

    Count
        Description: Count of occurrences (likely always 1; used for aggregation).
        Type: Integer

    Country, State, City
        Description: Geographical location of the customer.
        Type: String

    Zip Code
        Description: Postal code of the customer's location.
        Type: Integer

    Lat Long
        Description: Combined latitude and longitude information.
        Type: String

    Latitude, Longitude
        Description: Geographical coordinates of the customer.
        Type: Float

    Gender
        Description: Customer's gender (Male/Female).
        Type: String

    Senior Citizen
        Description: Indicates if the customer is a senior citizen (Yes/No).
        Type: binary 

    Partner
        Description: Indicates if the customer has a partner (Yes/No).
        Type: binary 

    Dependents
        Description: Indicates if the customer has dependents (Yes/No).
        Type: binary 

    Tenure Months
        Description: Number of months the customer has been with the company.
        Type: Integer

    Phone Service
        Description: Indicates if the customer has phone service (Yes/No).
        Type: binary 

    Multiple Lines
        Description: Indicates if the customer has multiple lines (Yes/No/No phone service).
        Type: String

    Internet Service
        Description: Type of internet service (DSL, Fiber optic, No).
        Type: String

    Online Security
        Description: Indicates if the customer subscribes to online security service (Yes/No/No internet service).
        Type: String

    Online Backup
        Description: Indicates if the customer subscribes to online backup service (Yes/No/No internet service).
        Type: String

    Device Protection
        Description: Indicates if the customer subscribes to device protection plan (Yes/No/No internet service).
        Type: String

    Tech Support
        Description: Indicates if the customer subscribes to tech support (Yes/No/No internet service).
        Type: String

    Streaming TV
        Description: Indicates if the customer subscribes to streaming TV service (Yes/No/No internet service).
        Type: String

    Streaming Movies
        Description: Indicates if the customer subscribes to streaming movies service (Yes/No/No internet service).
        Type: String

    Contract
        Description: Type of customer contract (Month-to-month, One year, Two year).
        Type: String

    Paperless Billing
        Description: Indicates if the customer has paperless billing (Yes/No).
        Type: binary 

    Payment Method
        Description: Customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card).
        Type: String

    Monthly Charges
        Description: The amount charged to the customer monthly.
        Type: Float

    Total Charges
        Description: The total amount charged to the customer.
        Type: Float

    Churn Label
        Description: Indicates if the customer has churned (Yes/No).
        Type: binary 

    Churn Value
        Description: Numeric representation of churn (1 for churned, 0 for not churned).
        Type: Integer

    Churn Score
        Description: A score indicating the likelihood of churn.
        Type: Integer

    CLTV
        Description: Customer Lifetime Value, a projection of the net profit attributed to the entire future relationship with the customer.
        Type: Integer

    Churn Reason
        Description: Reason for customer's churn (if applicable).
        Type: String

Data Quality and Preliminary Findings

    Data Cleaning: Handled missing values in 'Total Charges'.
    Key Insights:
        Shorter tenure and higher monthly charges are associated with higher churn rates.
        Services like online security reduce churn risk.
        Demographic factors (like being a senior citizen) influence churn likelihood.
