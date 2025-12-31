import pandas as pd

def load_cases(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = (df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    return df


def build_datetime(df):
    df["datetime"] = pd.to_datetime(
        df["date_occ"], errors="coerce"
    )

    df = df[df["datetime"].notna()]
    df["datetime"] = df["datetime"].dt.date

    return df


def create_mo_text(df):

    if "weapon_desc" in df.columns:
        weapon = df["weapon_desc"]
    else:
        weapon = df["crm_cd_desc"]

    df["mo_text"] = (
        df["crm_cd_desc"].astype(str).str.lower().fillna("") + " " +
        weapon.astype(str).str.lower().fillna("") + " " +
        df["area_name"].astype(str).str.lower().fillna("")
    )

    return df


def clean_cases(df):
    wanted = [
        "dr_no",
        "datetime",
        "area_name",
        "crm_cd",
        "crm_cd_desc",
        "weapon_desc",
        "vict_age",
        "vict_sex",
        "mo_text",
    ]
    df = df.replace("nan", "unknown")
    cols = [c for c in wanted if c in df.columns]
    return df[cols].reset_index(drop=True)


def preprocess_cases(input_path: str, output_path: str) -> None:
    df = load_cases(input_path)
    df = build_datetime(df)
    df = create_mo_text(df)
    df = clean_cases(df)
    df.to_csv(output_path, index=False)
