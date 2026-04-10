import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df[df["days_b_screening_arrest"].between(-30, 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    return df


def compute_metrics(df, race):
    group = df[df["race"] == race].copy()

    group["predicted"] = group["decile_score"] >= 5
    group["actual"] = group["two_year_recid"] == 1

    tp = ((group["predicted"] == 1) & (group["actual"] == 1)).sum()
    fp = ((group["predicted"] == 1) & (group["actual"] == 0)).sum()
    fn = ((group["predicted"] == 0) & (group["actual"] == 1)).sum()
    tn = ((group["predicted"] == 0) & (group["actual"] == 0)).sum()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return fpr, fnr


def main():
    df = load_data("data/compas-scores-two-years.csv")
    df = clean_data(df)

    for race in ["African-American", "Caucasian"]:
        fpr, fnr = compute_metrics(df, race)
        print(f"{race} -> FPR: {fpr:.3f}, FNR: {fnr:.3f}")


if __name__ == "__main__":
    main()