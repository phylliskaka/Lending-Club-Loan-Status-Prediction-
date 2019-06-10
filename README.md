# Lending-Club-Loan-Status-Prediction-
Developing a machine learning software package for predicting loan status 

## Project Goal 
## Dataset
These Dataset contain complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The dataset containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. It including 24 columns:  

1. addr_state: 2-letter code for the USA state of residence of the loan applicant.   
2. annual_inc: Annual income of the loan applicant.   
3. collections_12_mths_ex_med: Number of debt collections against the loan applicant in the 12 months previous to the loan inception.   
4. debt_to_income: Ratio of debt to income.  
5. delinq_2yrs: Number of times the loan applicant has missed a loan repayment during the past 2 years.  
6. revol_util: Loan applicant’s percentage utilization of their revolving credit facility, rounded to one decimal place.    
7. emp_length: Applicant’s length of time with current employer, in years.    
8. total_acc: Total number of accounts for the loan applicant. 
9. home_ownership: Whether the loan applicant owns, rents, or has a mortgage on their home.    
10. Id: Database row ID of the loan applicant 
11. initial_list_status: Whether the data is for a whole loan (vs. a fractional).    
12. inq_last_6mths: Credit enquiries about the applicant during the past 6 months.     
13. is_bad(target): Whether the loan defaulted or payments were missed.   
14. mths_since_last_delinq: Number of months since the load applicant last missed a loan repayment.   
15. mths_since_last_major_derog: Months since the last time seriously negative derogatory information was placed on the applicant’s credit record.   
16. mths_since_last_record: Number of months since the loan applicant’s last public record court judgement.     
17. zip_code: l 3-digit zip code of the applicant’s residential address.   
18. open_acc: Number of accounts the loan applicant has opened.     
19. pymnt_plan: Whether the loan applicant has been placed on a payment plan to bring their existing loans back to current status.   
20. policy_code: Which version of Lending Club’s lending criteria is applied.   
21. pub_rec: The number of public record judgements against the loan applicant.   
22. verification_status: Whether the income source is verified.    
23. purpose_cat: Purpose category for the loan.  
24. revol_bal: Balance on the loan applicant’s revolving credit facility.    

## Data Preprocessing 
1. Data Cleaning: fill null value and drop irrelevant data.
2. Explotory Data Analysis.
3. Principle Component Analysis 
4. Oversample the minor class in the dataset  

## Model Prediction 
Logistic regression 

## Result
Achieving 93% testing accuracy 
