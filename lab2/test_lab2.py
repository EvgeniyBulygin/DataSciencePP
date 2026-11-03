import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from correlation_tasks import FEATURES, analyze_brain_correlations
from pandas_tasks import analyze_titanic


def test_titanic_real_file():
    r = analyze_titanic(TitanicInput("titanic.csv"))
    assert r.row_count == 891
    assert r.missing_by_column["Age"] == 177
    assert r.missing_by_column["Cabin"] == 687
    assert r.missing_by_column["Embarked"] == 2
    assert all(type(v) is int for v in r.missing_by_column.values())
    assert set(r.mean_age_by_pclass) == {1, 2, 3}
    assert all(type(k) is int for k in r.mean_age_by_pclass)
    assert all(type(k) is int for k in r.survival_rate_by_pclass)
    assert abs(r.survival_rate_by_pclass[1] - 136 / 216) < 1e-12
    assert len(r.highest_fares) == 5
    assert r.highest_fares == sorted(r.highest_fares, reverse=True)
    assert r.highest_fares[0] == 512.3292


def test_titanic_small_file(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(
        "PassengerId,Survived,Pclass,Age,Fare\n"
        "1,1,1,40,10\n2,0,1,,20\n3,1,2,30,30\n4,0,2,31,40\n"
        "5,0,3,20,50\n6,1,3,,60\n7,0,3,50,70\n"
    )
    r = analyze_titanic(TitanicInput(str(csv)))
    assert r.row_count == 7
    assert r.missing_by_column["Age"] == 2
    assert r.adults_over_30_count == 3            # 40, 31, 50; ровно 30 не считается
    assert r.mean_age_by_pclass == {1: 40.0, 2: 30.5, 3: 35.0}   # NaN не стал нулём
    assert r.survival_rate_by_pclass == {1: 0.5, 2: 0.5, 3: 1 / 3}
    assert r.highest_fares == [70.0, 60.0, 50.0, 40.0, 30.0]


def test_brain_real_file():
    r = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
    df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])
    assert r.men_count == (df.Gender == "Male").sum() == 20
    assert r.women_count == (df.Gender == "Female").sum() == 20
    assert set(r.men_mri_correlation) == set(FEATURES) == set(r.women_mri_correlation)
    # независимая проверка через scipy (с ручным удалением пропусков)
    for gender, result in [("Male", r.men_mri_correlation), ("Female", r.women_mri_correlation)]:
        g = df[df.Gender == gender]
        for f in FEATURES:
            pair = g[[f, "MRI_Count"]].dropna()
            assert np.isclose(result[f], pearsonr(pair[f], pair["MRI_Count"])[0])
    both = {**{("m", k): v for k, v in r.men_mri_correlation.items()},
            **{("w", k): v for k, v in r.women_mri_correlation.items()}}
    best = max(both, key=lambda k: abs(both[k]))
    assert r.strongest_mri_feature == best[1]
    assert r.strongest_mri_feature != "MRI_Count"
    assert not any(np.isnan(v) for v in both.values())  # NA не ломают расчёт


def test_brain_strongest_uses_abs(tmp_path):
    # у мужчин Height идеально отрицательно коррелирует с MRI_Count
    f = tmp_path / "b.txt"
    f.write_text(
        "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
        "Male\t100\t100\t100\t150\t70\t1000\n"
        "Male\t101\t99\t102\t160\t68\t1100\n"
        "Male\t99\t102\t98\tNA\t66\t1200\n"
        "Male\t102\t100\t101\t155\t64\t1300\n"
        "Female\t100\t100\t100\t120\t60\t900\n"
        "Female\t105\t98\t99\t125\t61\t950\n"
        "Female\t98\t101\t103\t130\t63\t990\n"
        "Female\t101\t99\t100\t128\t62\t1020\n"
    )
    r = analyze_brain_correlations(BrainDataInput(str(f)))
    assert r.men_count == 4 and r.women_count == 4
    assert np.isclose(r.men_mri_correlation["Height"], -1.0)
    assert r.strongest_mri_feature == "Height"
