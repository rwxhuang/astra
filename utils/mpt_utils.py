def df_columns_mapping(df):
    '''
    given a dataframe df
    get a dictionary mapping of col names to their list of values
    '''
    return {col: df[col] for col in df.columns.to_list()}


def create_lambda_function(formula):
    '''
    formula: string representation of a formula
    return a lambda function that takes in a dictionary mapping
    the variable names to a list of their column values
    '''

    def lambda_function(**variables):
        '''
        variables: dictionary mapping var name to
        a list of their values
        '''
        return eval(formula, {}, variables)

    return lambda_function
