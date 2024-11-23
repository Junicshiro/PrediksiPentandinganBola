import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk mempersiapkan data
def prepare_data(matches):
    # Preprocessing yang sama seperti di notebook sebelumnya
    matches["date"] = pd.to_datetime(matches["date"])
    matches["target"] = (matches["result"] == "W").astype("int")
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    
    # Rolling averages
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    matches_rolling = matches.groupby("team").apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    return matches_rolling

# Fungsi rolling averages (sama seperti di notebook)
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Fungsi untuk melatih model
def train_model(matches_rolling):
    predictors = ["venue_code", "opp_code", "hour", "day_code", 
                  "gf_rolling", "ga_rolling", "sh_rolling", "sot_rolling", 
                  "dist_rolling", "fk_rolling", "pk_rolling", "pkatt_rolling"]
    
    train = matches_rolling[matches_rolling["date"] < '2022-01-01']
    
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train["target"])
    
    return rf, predictors

# Fungsi untuk memprediksi
def predict_match(model, predictors, team1_data, team2_data):
    # Gabungkan fitur untuk prediksi
    prediction_data = pd.DataFrame({
        'venue_code': [team1_data['venue_code']],
        'opp_code': [team2_data['opp_code']],
        'hour': [team1_data['hour']],
        'day_code': [team1_data['day_code']],
        'gf_rolling': [team1_data['gf_rolling']],
        'ga_rolling': [team1_data['ga_rolling']],
        'sh_rolling': [team1_data['sh_rolling']],
        'sot_rolling': [team1_data['sot_rolling']],
        'dist_rolling': [team1_data['dist_rolling']],
        'fk_rolling': [team1_data['fk_rolling']],
        'pk_rolling': [team1_data['pk_rolling']],
        'pkatt_rolling': [team1_data['pkatt_rolling']]
    })
    
    prediction = model.predict_proba(prediction_data)
    return prediction[0][1]  # Probabilitas menang

# Aplikasi Streamlit
def main():
    st.title('Prediksi Pertandingan Sepak Bola')
    
    # Muat data
    matches = pd.read_csv("matches.csv", index_col=0)
    
    # Siapkan data
    matches_rolling = prepare_data(matches)
    
    # Latih model
    model, predictors = train_model(matches_rolling)
    
    # Daftar tim
    teams = sorted(matches['team'].unique())
    
    # Pilih tim
    st.sidebar.header('Pilih Tim')
    team1 = st.sidebar.selectbox('Tim Pertama', teams)
    team2 = st.sidebar.selectbox('Tim Kedua', teams)
    
    # Ambil data terakhir untuk setiap tim
    team1_data = matches_rolling[matches_rolling['team'] == team1].iloc[-1]
    team2_data = matches_rolling[matches_rolling['team'] == team2].iloc[-1]
    
    # Prediksi
    if st.sidebar.button('Prediksi Pertandingan'):
        win_prob = predict_match(model, predictors, team1_data, team2_data)
        
        st.write(f"Probabilitas {team1} menang: {win_prob:.2%}")
        st.write(f"Probabilitas {team2} menang: {1-win_prob:.2%}")
        
        if win_prob > 0.5:
            st.success(f"Prediksi: {team1} akan menang!")
        else:
            st.success(f"Prediksi: {team2} akan menang!")

# Jalankan aplikasi
if __name__ == '__main__':
    main()