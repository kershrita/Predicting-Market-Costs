import pandas as pd

def set_index(df):
    df = (
        df
        .rename(columns={"Unnamed: 0": "id"})
        .set_index("id")
    )
    return df


def split_person_description(df):
    
    df[["personal", "deg_work"]] = (
        df["Person Description"]
        .str
        .split(", education: ", expand=True)
    )
    
    df[["Marriage", "Gender", "with", "Children", "tc"]] = (
        df["personal"]
        .str
        .split(expand=True)
    )
    
    df[["Degree", "Work"]] = (
        df["deg_work"]
        .str
        .split("working as", expand=True)
    )
    
    df = df.drop(columns=["Person Description", "personal", "with", "tc", "deg_work"])
    return df


def split_place_code(df):
    df[["Store Code", "Country ISO2"]] = (
        df["Place Code"]
        .str
        .split("_", expand=True)
    )
    
    df = df.drop(columns="Place Code")
    return df


def split_customer_order(df):
    df[["ord_dep", "Oreder Brand"]] = (
        df["Customer Order"]
        .str
        .split(", Ordered Brand : ", expand=True)
    )
    
    df[["Order", "Department", "blank"]] = (
        df["ord_dep"]
        .str
        .split("from | department", expand=True)
    )
        
    df = df.drop(columns=["Customer Order", "ord_dep", "blank"])
    return df


def encode_market_features(df):
    unique_feat = set()
    
    for feat_list in df["Additional Features in market"] :
        if pd.notna(feat_list):
            string_data = feat_list.strip("[]")
            elements = string_data.split(', ')
            elements = [element.strip("'") for element in elements]
            unique_feat.update(elements)

    for feat in unique_feat:
        df[feat] = (
            df["Additional Features in market"]
            .apply(lambda x: 1 if pd.notna(x) and feat in x else 0)
        )
        
    df = df.drop(columns="Additional Features in market")
    return df


def transform_cost_sales(df):
    df["Store Sales"] = (
        df["Store Sales"]
        .str
        .split(expand=True)[0]
        .astype(float)
    ) * 1e6
    
    df["Store Cost"] = (
        df["Store Cost"]
        .str
        .split(expand=True)[0]
        .astype(float)
    ) * 1e6
    return df


def extract_product_weights(df):
    if "Product Weights Data in (KG)" in df.columns:
        df[["b1", "Gross Weight", "Net Weight", "Package Weight", "b2"]] = (
            df["Product Weights Data in (KG)"]
            .str
            .split("{'Gross Weight': |, 'Net Weight': |, 'Package Weight': |}", expand=True)
        )
        df = df.drop(columns=["b1", "b2", "Product Weights Data in (KG)"])
    elif "Weights Data" in df.columns:
        df[["b1", "Gross Weight", "Net Weight", "Package Weight", "b2"]] = (
            df["Weights Data"]
            .str
            .split("{'Gross Weight': |, 'Net Weight': |, 'Package Weight': |}", expand=True)
        )
        df = df.drop(columns=["b1", "b2", "Weights Data"])
    return df


def transform_recyclable(df):
    mapping = {'recyclable': 'yes', 'non recyclable': 'no'}
    df["Is Recyclable?"] = df["Is Recyclable?"].map(mapping)
    return df


def transform_income(df):
    if "Min. Yearly Income" in df.columns:
        df["Min. Person Yearly Income"] = (
            df["Min. Yearly Income"]
            .str
            .split("K+", expand=True)[0]
            .astype(float) * 1000
        )
        df = df.drop(columns="Min. Yearly Income")
        
    elif "Min. Person Yearly Income" in df.columns:
        df["Min. Person Yearly Income"] = (
            df["Min. Person Yearly Income"]
            .str
            .split("K+", expand=True)[0]
            .astype(float) * 1000
        )
        
    elif "Yearly Income" in df.columns:
        df["Min. Person Yearly Income"] = (
            df["Yearly Income"]
            .str
            .split("K+", expand=True)[0]
            .astype(float) * 1000
        )
        df = df.drop(columns="Yearly Income")
    return df


def transform_columns_type(df):
    df["Store Area"] = (
        df["Store Area"]
        .replace('missing', float('nan'))
        .astype(float)
    )
    
    df["Grocery Area"] = (
        df["Grocery Area"]
        .str
        .strip('"')
        .replace('missing', float('nan'))
        .astype(float)
    )
    
    df["Meat Area"] = (
        df["Meat Area"]
        .str
        .strip('"')
        .astype(float)
    )
    
    trans = ["Gross Weight", "Net Weight", "Package Weight"]
    
    df[trans] = df[trans].astype(float)
    
    return df


def calculate_package_weight(df):
    df["Package Weight"] = df["Gross Weight"] - df["Net Weight"]
    return df


def fill_nulls(df):
    int_cols = df.select_dtypes([int, bool]).columns
    float_cols = df.select_dtypes(float).columns
    cat_cols = df.select_dtypes("object").columns
    
    if "Cost" in df.columns:
        df[float_cols] = (
            df[float_cols]
            .fillna(df[float_cols].mean())
        )
        
        for col in int_cols:
            df[col] = (
                df[col]
                .fillna(df[col].mode()[0])
            )
        
    else:
        df[float_cols] = (
            df[float_cols]
            .fillna(df[float_cols].mean())
        )
        
        for col in int_cols:
            df[col] = (
                df[col]
                .fillna(df[col].mode()[0])
            )
        
        for col in cat_cols:
            df[col] = (
                df[col]
                .fillna(df[col].mode()[0])
            )

    return df


def encode_columns(df):
    mapping = {'yes': 1, 'no': 0}
    df["Is Recyclable?"] = (
        df["Is Recyclable?"]
        .map(mapping)
        .astype(bool)
    )
    
    mapping = {'five': 5, 'four': 4, 'three': 3, 'two': 2, 'one': 1, 'no': 0}
    df["Children"] = (
        df["Children"]
        .map(mapping)
    )
    
    df["Children"] = (
        df["Children"]
        .fillna(df["Children"].mode()[0])
        .astype(int)
    )
    return df
