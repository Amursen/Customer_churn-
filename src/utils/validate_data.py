import great_expectations as ge
import  pandas as pd
from typing import  tuple , list



def validate_telco_data(df) -> bool[bool , list[str]]:

    ge_df = ge.dataset.PandasDataset(df)



    ge_df.expect_column_to_exist("customer ID")
    ge_df.expect_column_values_not_to_be_null("customer ID")



    ge_df.expect_column_to_exist("gender")
    ge_df.expect_colimn_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependets")




    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")


    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")


    ge_df.expect_column_values_to_be_in_set("gender" , ["Male" , "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner" , ["Yes" , "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents" , ["Yes" , "No"])
    ge_df.expect_column_values_to_be_in_set("Contarct" , ["Month-to-month" , "One year" , "Two year"])
    ge_df.expect_column_values_to_be_in_set("internetService" , ["DSL" , "fiber optic" , "No"])



    ge_df.expect_column_values_to_between("tenure" , min_value = 0 , max_value = 120)
    ge_df.expect_column_values_to_between("MonthlyCharge" , min_value = 0  , max_value = 200)
    ge_df.expect_column_values_to_between("TotalCharge" , min_value = 0)


    ge_df.expect_column_values_not_to_be_null("tenure")
    ge_df.expect_column_values_not_to_be_null("MonthlyCharge")






    ge_df.expect_column_pair_values_A_to_be_greater_than_B(column_A = "totalCharge" , column_B = "MonthlyCharge", or_equal = True , mostly = 0.95)





    results = ge_df.validate()




    failed_expectation = []

    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]

            failed_expectation.append(expectation_type)




    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks



    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectation}")

    return results["success"], failed_expectation
