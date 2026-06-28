from __future__ import annotations

import io, re, math, zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CKK Portfolio Builder v13.2", page_icon="🏆", layout="wide")

ROSTER_COLS = ["SP1","SP2","P","C","1B","2B","3B","SS","OF1","OF2","OF3","OF","UTIL","CPT","FLEX","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5","FLEX6","PG","SG","SF","PF","G"]

TEAM_ALIAS = {"CHW":"CWS","CWS":"CWS","KCR":"KC","KC":"KC","SDP":"SD","SFG":"SF","TBR":"TB","WSN":"WSH","WAS":"WSH","ATH":"OAK"}
def norm_team(x):
    s = str(x).upper().strip()
    return TEAM_ALIAS.get(s, s)

# ballpark, city label, lat, lon, roof flag. Roof parks get reduced weather risk.
TEAM_PARKS = {
"ARI":("Chase Field","Phoenix, AZ",33.4455,-112.0667,True),"ATL":("Truist Park","Atlanta, GA",33.8908,-84.4678,False),"BAL":("Camden Yards","Baltimore, MD",39.2840,-76.6217,False),"BOS":("Fenway Park","Boston, MA",42.3467,-71.0972,False),"CHC":("Wrigley Field","Chicago, IL",41.9484,-87.6553,False),"CWS":("Rate Field","Chicago, IL",41.8300,-87.6339,False),"CIN":("Great American Ball Park","Cincinnati, OH",39.0979,-84.5082,False),"CLE":("Progressive Field","Cleveland, OH",41.4962,-81.6852,False),"COL":("Coors Field","Denver, CO",39.7559,-104.9942,False),"DET":("Comerica Park","Detroit, MI",42.3390,-83.0485,False),"HOU":("Daikin Park","Houston, TX",29.7573,-95.3555,True),"KC":("Kauffman Stadium","Kansas City, MO",39.0517,-94.4803,False),"LAA":("Angel Stadium","Anaheim, CA",33.8003,-117.8827,False),"LAD":("Dodger Stadium","Los Angeles, CA",34.0739,-118.2400,False),"MIA":("loanDepot park","Miami, FL",25.7781,-80.2197,True),"MIL":("American Family Field","Milwaukee, WI",43.0280,-87.9712,True),"MIN":("Target Field","Minneapolis, MN",44.9817,-93.2776,False),"NYM":("Citi Field","Queens, NY",40.7571,-73.8458,False),"NYY":("Yankee Stadium","Bronx, NY",40.8296,-73.9262,False),"OAK":("Sutter Health Park","Sacramento, CA",38.5804,-121.5130,False),"PHI":("Citizens Bank Park","Philadelphia, PA",39.9061,-75.1665,False),"PIT":("PNC Park","Pittsburgh, PA",40.4469,-80.0057,False),"SD":("Petco Park","San Diego, CA",32.7076,-117.1570,False),"SF":("Oracle Park","San Francisco, CA",37.7786,-122.3893,False),"SEA":("T-Mobile Park","Seattle, WA",47.5914,-122.3325,True),"STL":("Busch Stadium","St. Louis, MO",38.6226,-90.1928,False),"TB":("Tropicana Field","St. Petersburg, FL",27.7682,-82.6534,True),"TEX":("Globe Life Field","Arlington, TX",32.7473,-97.0842,True),"TOR":("Rogers Centre","Toronto, ON",43.6414,-79.3894,True),"WSH":("Nationals Park","Washington, DC",38.8730,-77.0074,False)}

WEIGHTS = {
    "CKK": {"Win%":.25,"Finish_percentile":.20,"Lineup Edge":.15,"Geomean":.10,"Diversity":.10,"Weighted Own Fit":.10,"Duplication Score":.05,"median":.05},
    "FPE": {"Win%":.35,"Lineup Edge":.20,"Geomean":.15,"Finish_percentile":.10,"Duplication Score":.10,"Weighted Own Fit":.05,"Diversity":.05},
    "Confidence": {"Finish_percentile":.30,"Geomean":.20,"median":.15,"Lineup Edge Stability":.15,"Diversity":.10,"Weighted Own Fit":.05,"Duplication Score":.05},
}

def clean_cols(df):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]; return df

def read_csv(uploaded):
    data = uploaded.getvalue()
    for enc in ["utf-8", "latin1"]:
        try: return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception: pass
    return pd.read_csv(io.BytesIO(data), engine="python")

def classify_file(name: str, df: pd.DataFrame) -> str:
    cols = {str(c).strip().lower() for c in df.columns}
    name_l = name.lower()
    if {"game info","teamabbrev","salary"} <= cols: return "DK Salaries"
    if "matchups" in name_l or {"avg score","avg score l5","trending score"} & cols and {"team","opp"} <= cols: return "MLB Matchups Master"
    if {"names","runs","hr"} <= cols or "park_factors" in name_l: return "Park Factors"
    if {"xwoba","iso"} & cols and {"team"} <= cols and ({"hardhits","barrels","homeruns","strikeouts"} & cols): return "Team Gamelogs"
    # ROO / projection exports can use either player_names, name, Player, or player name.
    # Important: detect ROO BEFORE scoring sheets because ROO files include percentage columns like 15+%, 2x%, 3x%.
    # Common MLB ROO columns: Player, Position, Order, Team, Opp, Salary, Floor, Median, Ceiling, Ownership, Small Own, Large Own.
    has_player_col = any(c in cols for c in ["player_names", "player", "name", "player name"])
    has_projection_cols = {"position", "team", "salary"} <= cols and any(c in cols for c in ["median", "ceiling", "floor", "projection", "proj", "fpts"])
    has_ownership_cols = any("own" in c for c in cols) or "ownership" in cols
    if has_player_col and has_projection_cols and has_ownership_cols:
        return "ROO Projections"
    # Scoring sheets usually have score/scoring/runs-style team scoring columns, but not player/projection columns.
    is_scoring_by_name = "scoring" in name_l
    is_scoring_by_cols = ("team" in cols and ("score" in cols or any("score" in c for c in cols)) and any("8+" in c or "pct" in c or "%" in c for c in cols))
    if is_scoring_by_name or is_scoring_by_cols:
        return "Scoring %"
    roster_hit = len(set(c.upper() for c in df.columns) & set(ROSTER_COLS))
    pm_metrics = {"win%","finish_percentile","lineup edge","geomean","dupes","diversity","weighted own"}
    if roster_hit >= 4 or len(cols & pm_metrics) >= 3: return "Portfolio Manager"
    return "Unknown"

def pct_rank(s, higher=True):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0: return pd.Series(50.0, index=s.index)
    r = x.rank(pct=True)*100
    if not higher: r = 100-r
    return r.fillna(50).clip(0,100)

def weighted_own_fit(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum()==0: return pd.Series(50.0, index=s.index)
    q25,q75=x.quantile(.25),x.quantile(.75); iqr=max(q75-q25,1e-9); target=x.median()-.15*iqr
    return (100-(abs(x-target)/(2.25*iqr)*100)).fillna(50).clip(0,100)

def dupe_score(s):
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    return (100/(1+np.sqrt(np.maximum(x,0)))).clip(0,100)

def edge_stability(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum()==0: return pd.Series(50.0, index=s.index)
    return (pct_rank(x, True) - (abs(x-x.median())/(x.std(ddof=0)+1e-9)*6).clip(0,22)).fillna(50).clip(0,100)

def add_scores(df):
    out=clean_cols(df)
    for c in out.columns:
        if c not in ROSTER_COLS: out[c]=out[c]
    idx=out.index
    def get(c): return out[c] if c in out.columns else pd.Series(index=idx, dtype=float)
    norm=pd.DataFrame(index=idx)
    for c in ["Win%","Finish_percentile","Lineup Edge","Geomean","Diversity","median"]: norm[c]=pct_rank(get(c), True)
    norm["Weighted Own Fit"] = weighted_own_fit(out["Weighted Own"] if "Weighted Own" in out.columns else out["Own"] if "Own" in out.columns else pd.Series(index=idx,dtype=float))
    norm["Duplication Score"] = dupe_score(out["Dupes"] if "Dupes" in out.columns else pd.Series(0,index=idx))
    norm["Lineup Edge Stability"] = edge_stability(get("Lineup Edge"))
    for score, w in WEIGHTS.items():
        out[score] = sum(norm[k]*v for k,v in w.items()).round(2).clip(0,100)
    out["PBI"]=(.40*out["CKK"]+.35*out["FPE"]+.25*out["Confidence"]).round(2)
    out["Print Probability"]=(.42*out["FPE"]+.33*out["CKK"]+.15*out["Confidence"]+.10*norm["Duplication Score"]).round(2).clip(0,100)
    out["Decision"] = np.select([(out.CKK>=85)&(out.FPE>=80)&(out.Confidence>=75),(out.CKK>=72)&(out.FPE>=70),(out.CKK>=60)|(out.FPE>=65)], ["🟢 Play","🟡 Strong GPP","🟠 Fringe"], "🔴 Rebuild")
    return out

def parse_game_info(value: str) -> Tuple[Optional[str],Optional[str],str]:
    s=str(value).upper().strip()
    # DK usually: PHI@NYM 07/01/2026 07:10PM ET
    m=re.search(r'([A-Z]{2,3})\s*@\s*([A-Z]{2,3})', s)
    if not m: return None,None,"Unknown"
    away,home=norm_team(m.group(1)),norm_team(m.group(2))
    return away,home,f"{away} @ {home}"

def build_game_map(dk: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dk is None or "Game Info" not in dk.columns: return pd.DataFrame()
    rows=[]
    for gi in dk["Game Info"].dropna().unique():
        away,home,matchup=parse_game_info(gi)
        if home:
            park,city,lat,lon,roof=TEAM_PARKS.get(home,("Unknown","Unknown",np.nan,np.nan,False))
            rows.append({"Matchup":matchup,"Away":away,"Home":home,"Game Info":gi,"Ballpark":park,"City":city,"Lat":lat,"Lon":lon,"Roof/Potential Roof":roof})
    return pd.DataFrame(rows).drop_duplicates("Matchup").sort_values("Matchup") if rows else pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> dict:
    if pd.isna(lat) or pd.isna(lon): return {}
    url="https://api.open-meteo.com/v1/forecast"
    params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m","forecast_days":2,"temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"auto"}
    try:
        r=requests.get(url, params=params, timeout=8); r.raise_for_status(); return r.json()
    except Exception: return {}

def summarize_weather(game_map: pd.DataFrame, manual_weather: Optional[pd.DataFrame], auto_fetch: bool) -> pd.DataFrame:
    if game_map.empty: return pd.DataFrame()
    rows=[]
    manual = clean_cols(manual_weather) if manual_weather is not None else None
    for _,g in game_map.iterrows():
        temp=72; wind=5; rain=0; source="API off / default"
        if manual is not None and not manual.empty:
            cols={c.lower():c for c in manual.columns}
            team_col=next((cols[c] for c in cols if c in ["team","home","home team","teamabbrev"]), None)
            match_col=next((cols[c] for c in cols if c in ["matchup","game","game info"]), None)
            sub=manual.iloc[0:0]
            if team_col: sub=manual[manual[team_col].astype(str).str.upper().map(norm_team).eq(g.Home)]
            if sub.empty and match_col: sub=manual[manual[match_col].astype(str).str.upper().str.contains(str(g.Away),na=False)&manual[match_col].astype(str).str.upper().str.contains(str(g.Home),na=False)]
            if not sub.empty:
                row=sub.iloc[0]
                def pick(names, default):
                    for n in names:
                        for c in manual.columns:
                            if n in c.lower():
                                val=pd.to_numeric(row[c], errors="coerce")
                                if pd.notna(val): return float(val)
                    return default
                temp=pick(["temp","temperature"],temp); wind=pick(["wind speed","wind"],wind); rain=pick(["precip","rain","delay"],rain); source="uploaded weather"
        elif auto_fetch:
            data=fetch_weather(float(g.Lat), float(g.Lon))
            hr=data.get("hourly",{}) if data else {}
            if hr and hr.get("temperature_2m"):
                # use max rain/wind and avg temp over next 12 forecast points as a slate approximation
                temp=float(np.nanmean(hr.get("temperature_2m",[])[:12] or [72])); wind=float(np.nanmax(hr.get("wind_speed_10m",[])[:12] or [5])); rain=float(np.nanmax(hr.get("precipitation_probability",[])[:12] or [0])); source="Open-Meteo API next 12h"
        roof=bool(g["Roof/Potential Roof"])
        raw_risk = rain*.65 + max(wind-12,0)*2.0 + (8 if wind>=18 else 0)
        if roof: raw_risk *= .25
        risk=float(np.clip(raw_risk,0,100))
        label="🟢 Low"
        if risk>=55: label="🔴 High"
        elif risk>=25: label="🟡 Moderate"
        hitter_boost=np.clip((temp-70)*.45 + max(wind-10,0)*.7 - risk*.15, -20, 20)
        rows.append({"Matchup":g.Matchup,"Away":g.Away,"Home":g.Home,"Teams":f"{g.Away}, {g.Home}","Ballpark":g.Ballpark,"City":g.City,"Roof/Potential Roof":roof,"Temp °F":round(temp,1),"Wind mph":round(wind,1),"Rain/Delay %":round(rain,1),"Weather Risk":round(risk,1),"Risk Label":label,"Hitter Weather Boost":round(hitter_boost,1),"Weather Source":source})
    return pd.DataFrame(rows)

def team_from_stack(stack: str) -> Optional[str]:
    s=str(stack).upper().strip()
    # Portfolio Manager Stack often starts with team abbrev; find any known team token.
    for token in re.split(r'[^A-Z]+', s):
        t=norm_team(token)
        if t in TEAM_PARKS: return t
    if len(s)>=2:
        t=norm_team(s[:3]);
        if t in TEAM_PARKS: return t
    return None

def park_score_table(park_df: Optional[pd.DataFrame]) -> Dict[str,float]:
    if park_df is None: return {}
    p=clean_cols(park_df)
    name_col=next((c for c in p.columns if c.lower() in ["names","name","ballpark","park"]),None)
    if not name_col: return {}
    nums=[]
    for c in p.columns:
        if c!=name_col:
            vals=pd.to_numeric(p[c], errors="coerce")
            if vals.notna().sum()>0: p[c]=vals; nums.append(c)
    if not nums: return {}
    p["_park_score"] = p[nums].mean(axis=1).rank(pct=True)*100
    return dict(zip(p[name_col].astype(str), p["_park_score"].round(1)))

def team_metric_scores(df: Optional[pd.DataFrame], team_col_candidates=("Team","team")) -> Dict[str,float]:
    if df is None: return {}
    t=clean_cols(df)
    team_col=next((c for c in t.columns if c.lower() in [x.lower() for x in team_col_candidates]), None)
    if not team_col: return {}
    t[team_col]=t[team_col].map(norm_team)
    good_keys=["xwoba","iso","hard","barrel","homer","avg score","trending","8+","runs","score"]
    bad_keys=["strikeout","k%"]
    components=[]
    for c in t.columns:
        if c==team_col: continue
        vals=pd.to_numeric(t[c], errors="coerce")
        if vals.notna().sum()==0: continue
        lc=c.lower()
        if any(k in lc for k in good_keys): components.append(pct_rank(vals, True))
        elif any(k in lc for k in bad_keys): components.append(pct_rank(vals, False))
    if not components: return {}
    t["_score"]=pd.concat(components, axis=1).mean(axis=1)
    return t.groupby(team_col)["_score"].mean().round(1).to_dict()

def add_mlb_context(portfolio: pd.DataFrame, game_map: pd.DataFrame, weather: pd.DataFrame, park_df, team_df, scoring_df, master_df):
    out=portfolio.copy()
    out["Primary Stack Team"] = out["Stack"].map(team_from_stack) if "Stack" in out.columns else None
    # If Stack missing, try most common roster team via DKSalaries is too expensive; keep Unknown.
    team_to_game={}
    for _,g in game_map.iterrows():
        team_to_game[g.Away]=g.to_dict(); team_to_game[g.Home]=g.to_dict()
    for col in ["Matchup","Ballpark","Home/Away","Weather Label"]:
        out[col] = "Unknown"
    for col in ["Weather Risk","Weather Boost","Weather Score","Park Score","Team/Master Trend Score","MLB Stack Score"]:
        out[col] = 0.0
    park_scores=park_score_table(park_df); team_scores=team_metric_scores(team_df); scoring_scores=team_metric_scores(scoring_df); master_scores=team_metric_scores(master_df, ("Team",))
    stack_scores=[]; park_s=[]; trend_s=[]; weather_s=[]
    for i,row in out.iterrows():
        tm=row.get("Primary Stack Team")
        gm=team_to_game.get(tm,{})
        if gm:
            out.at[i,"Matchup"]=gm.get("Matchup","Unknown"); out.at[i,"Ballpark"]=gm.get("Ballpark","Unknown"); out.at[i,"Home/Away"]="Home" if tm==gm.get("Home") else "Away"
        wrow=weather[weather["Matchup"].eq(out.at[i,"Matchup"])] if not weather.empty else pd.DataFrame()
        wrisk=float(wrow["Weather Risk"].iloc[0]) if not wrow.empty else 0.0
        wboost=float(wrow["Hitter Weather Boost"].iloc[0]) if not wrow.empty else 0.0
        out.at[i,"Weather Risk"]=round(wrisk,1); out.at[i,"Weather Label"]=(wrow["Risk Label"].iloc[0] if not wrow.empty else "Unknown"); out.at[i,"Weather Boost"]=round(wboost,1)
        ps=park_scores.get(out.at[i,"Ballpark"],50); ts=np.nanmean([team_scores.get(tm,50), scoring_scores.get(tm,50), master_scores.get(tm,50)])
        ws=np.clip(100-wrisk+wboost,0,100)
        park_s.append(ps); trend_s.append(ts); weather_s.append(ws)
        base=np.nanmean([row.get("CKK",50), row.get("FPE",50), ps, ts, ws])
        stack_scores.append(round(float(base),2))
    out["Park Score"]=park_s; out["Team/Master Trend Score"]=np.round(trend_s,1); out["Weather Score"]=np.round(weather_s,1); out["MLB Stack Score"]=stack_scores
    out["MLB CKK"]=(.80*out["CKK"]+.12*out["MLB Stack Score"]+.08*out["Weather Score"]).round(2).clip(0,100)
    return out


def grade_from_score(x):
    try: x=float(x)
    except Exception: return "N/A"
    if x>=97: return "A+"
    if x>=93: return "A"
    if x>=90: return "A-"
    if x>=87: return "B+"
    if x>=83: return "B"
    if x>=80: return "B-"
    if x>=75: return "C+"
    if x>=70: return "C"
    if x>=65: return "C-"
    if x>=60: return "D"
    return "F"

def status_from_score(x, hot_label="Hot", cold_label="Cold"):
    try: x=float(x)
    except Exception: return "Unknown"
    if x>=80: return "🔥 " + hot_label
    if x<=45: return "❄️ " + cold_label
    return "⚪ Neutral"

def find_col(df, options):
    if df is None: return None
    cols={str(c).lower().strip():c for c in df.columns}
    for o in options:
        if o.lower() in cols: return cols[o.lower()]
    for c in df.columns:
        lc=str(c).lower()
        if any(o.lower() in lc for o in options): return c
    return None

def normalize_player_df(roo):
    if roo is None: return pd.DataFrame()
    r=clean_cols(roo).copy()
    ren={}
    player=find_col(r,["player_names","player","name","player name"])
    pos=find_col(r,["position","pos"])
    teamc=find_col(r,["team","teamabbrev"])
    salary=find_col(r,["salary"])
    median=find_col(r,["median","projection","proj","fpts"])
    ceiling=find_col(r,["ceiling","ceil"])
    own=find_col(r,["ownership","own","large own","small own"])
    order=find_col(r,["order","batting order","lineup"])
    for old,new in [(player,"Player"),(pos,"Position"),(teamc,"Team"),(salary,"Salary"),(median,"Median"),(ceiling,"Ceiling"),(own,"Ownership"),(order,"Order")]:
        if old: ren[old]=new
    r=r.rename(columns=ren)
    for c in ["Player","Position","Team","Salary","Median","Ceiling","Ownership","Order"]:
        if c not in r.columns: r[c]=np.nan
    r["Team"]=r["Team"].map(norm_team)
    r["Position"]=r["Position"].astype(str).str.upper()
    for c in ["Salary","Median","Ceiling","Ownership","Order"]: r[c]=pd.to_numeric(r[c], errors="coerce")
    return r

def make_hitter_trends(roo, game_map, weather, team_df, scoring_df, master_df):
    p=normalize_player_df(roo)
    if p.empty: return pd.DataFrame()
    hitters=p[~p["Position"].str.contains("P", na=False)].copy()
    if hitters.empty: return hitters
    team_scores=team_metric_scores(team_df); scoring_scores=team_metric_scores(scoring_df); master_scores=team_metric_scores(master_df,("Team",))
    team_to_weather={}
    if not weather.empty:
        for _,w in weather.iterrows():
            for t in [w.get("Away"), w.get("Home")]: team_to_weather[t]=w.to_dict()
    hitters["Proj Score"]=pct_rank(hitters["Median"], True)
    hitters["Ceiling Score"]=pct_rank(hitters["Ceiling"], True)
    hitters["Value Score"]=pct_rank(hitters["Median"]/(hitters["Salary"].replace(0,np.nan)/1000), True)
    hitters["Own Fit"]=weighted_own_fit(hitters["Ownership"])
    hitters["Team Trend"]=[np.nanmean([team_scores.get(t,50), scoring_scores.get(t,50), master_scores.get(t,50)]) for t in hitters["Team"]]
    hitters["Weather Boost"]=[team_to_weather.get(t,{}).get("Hitter Weather Boost",0) for t in hitters["Team"]]
    hitters["Hitter Trend Score"]=(.22*hitters["Proj Score"]+.20*hitters["Ceiling Score"]+.18*hitters["Value Score"]+.16*hitters["Own Fit"]+.16*hitters["Team Trend"]+.08*(50+hitters["Weather Boost"].fillna(0))).round(1).clip(0,100)
    hitters["Hitter Trend Grade"]=hitters["Hitter Trend Score"].map(grade_from_score)
    hitters["Ceiling Rating"]=hitters["Ceiling Score"].map(grade_from_score)
    hitters["Value Rating"]=hitters["Value Score"].map(grade_from_score)
    hitters["Boom Probability"]=(.55*hitters["Hitter Trend Score"]+.45*hitters["Ceiling Score"]).round(1).clip(0,100)
    hitters["Why Today"]=("Trend " + hitters["Hitter Trend Grade"].astype(str) + "; ceiling " + hitters["Ceiling Rating"].astype(str) + "; value " + hitters["Value Rating"].astype(str) + "; team/weather context included")
    return hitters.sort_values("Hitter Trend Score", ascending=False)

def make_pitcher_trends(roo, game_map, weather, team_df, scoring_df, master_df):
    p=normalize_player_df(roo)
    if p.empty: return pd.DataFrame()
    pit=p[p["Position"].str.contains("P", na=False)].copy()
    if pit.empty: return pit
    pit["Proj Score"]=pct_rank(pit["Median"], True)
    pit["Ceiling Score"]=pct_rank(pit["Ceiling"], True)
    pit["Value Score"]=pct_rank(pit["Median"]/(pit["Salary"].replace(0,np.nan)/1000), True)
    # Pitcher ownership fit rewards strong pitchers but slightly avoids extreme chalk.
    pit["Own Fit"]=weighted_own_fit(pit["Ownership"])
    pit["Pitcher Trend Score"]=(.30*pit["Ceiling Score"]+.28*pit["Proj Score"]+.17*pit["Value Score"]+.15*pit["Own Fit"]+.10*50).round(1).clip(0,100)
    pit["Pitcher Trend Grade"]=pit["Pitcher Trend Score"].map(grade_from_score)
    pit["Hot/Neutral/Cold"]=pit["Pitcher Trend Score"].map(lambda x: status_from_score(x,"Hot","Cold"))
    pit["Risk Meter"]=(100-(.55*pit["Pitcher Trend Score"]+.45*pit["Own Fit"])).round(1).clip(0,100)
    pit["Ceiling Meter"]=pit["Ceiling Score"].round(1)
    pit["Bust Probability"]=(100-(.45*pit["Pitcher Trend Score"]+.35*pit["Proj Score"]+.20*pit["Value Score"])).round(1).clip(0,100)
    pit["Why Today"]=("Pitcher trend " + pit["Pitcher Trend Grade"].astype(str) + "; ceiling meter " + pit["Ceiling Meter"].astype(str) + "; ownership/value adjusted")
    return pit.sort_values("Pitcher Trend Score", ascending=False)


MLB_TEAM_NAMES = {
    "ARI":["ARIZONA DIAMONDBACKS","DIAMONDBACKS"],"ATL":["ATLANTA BRAVES","BRAVES"],"BAL":["BALTIMORE ORIOLES","ORIOLES"],"BOS":["BOSTON RED SOX","RED SOX"],
    "CHC":["CHICAGO CUBS","CUBS"],"CWS":["CHICAGO WHITE SOX","WHITE SOX"],"CIN":["CINCINNATI REDS","REDS"],"CLE":["CLEVELAND GUARDIANS","GUARDIANS"],
    "COL":["COLORADO ROCKIES","ROCKIES"],"DET":["DETROIT TIGERS","TIGERS"],"HOU":["HOUSTON ASTROS","ASTROS"],"KC":["KANSAS CITY ROYALS","ROYALS"],
    "LAA":["LOS ANGELES ANGELS","ANGELS"],"LAD":["LOS ANGELES DODGERS","DODGERS"],"MIA":["MIAMI MARLINS","MARLINS"],"MIL":["MILWAUKEE BREWERS","BREWERS"],
    "MIN":["MINNESOTA TWINS","TWINS"],"NYM":["NEW YORK METS","METS"],"NYY":["NEW YORK YANKEES","YANKEES"],"OAK":["ATHLETICS","OAKLAND ATHLETICS","A'S"],
    "PHI":["PHILADELPHIA PHILLIES","PHILLIES"],"PIT":["PITTSBURGH PIRATES","PIRATES"],"SD":["SAN DIEGO PADRES","PADRES"],"SF":["SAN FRANCISCO GIANTS","GIANTS"],
    "SEA":["SEATTLE MARINERS","MARINERS"],"STL":["ST. LOUIS CARDINALS","ST LOUIS CARDINALS","CARDINALS"],"TB":["TAMPA BAY RAYS","RAYS"],
    "TEX":["TEXAS RANGERS","RANGERS"],"TOR":["TORONTO BLUE JAYS","BLUE JAYS"],"WSH":["WASHINGTON NATIONALS","NATIONALS"]
}

def american_to_prob(odds):
    try: odds=float(odds)
    except Exception: return np.nan
    if odds < 0: return abs(odds)/(abs(odds)+100)
    return 100/(odds+100)

def estimate_implied_runs(total, team_prob):
    # Lightweight approximation: total split based on win probability.
    try:
        total=float(total); team_prob=float(team_prob)
    except Exception:
        return np.nan
    adj=(team_prob-.50)*1.6
    return max(0, total/2 + adj)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_oddsapi_mlb(api_key: str):
    """Return (events, status_message). Uses The Odds API current MLB odds.
    If the key is missing/invalid or the request fails, the app will mark Vegas as Not Connected
    instead of pretending neutral 50s are real Vegas scores.
    """
    if not api_key:
        return [], "No API key entered"
    url="https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params={"apiKey":api_key,"regions":"us","markets":"h2h,totals","oddsFormat":"american"}
    try:
        r=requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return [], f"Odds API error {r.status_code}: {r.text[:160]}"
        data=r.json()
        return data, f"Connected: {len(data)} MLB odds events returned"
    except Exception as e:
        return [], f"Odds API request failed: {e}"

def team_name_match(name: str) -> Optional[str]:
    n=str(name).upper().replace(".","").strip()
    for abbr,names in MLB_TEAM_NAMES.items():
        if any(x.replace(".","") in n or n in x.replace(".","") for x in names):
            return abbr
    return None

def build_vegas_table(game_map: pd.DataFrame, api_key: str = "") -> pd.DataFrame:
    if game_map is None or game_map.empty:
        return pd.DataFrame()
    odds, status = fetch_oddsapi_mlb(api_key)
    rows=[]
    # IMPORTANT: no fake 50s as real Vegas. Missing odds are clearly Not Connected and excluded from Stack Score weights.
    for _,g in game_map.iterrows():
        rows.append({"Matchup":g.Matchup,"Away":g.Away,"Home":g.Home,"Game Total":np.nan,"Away ML":np.nan,"Home ML":np.nan,"Away ITT":np.nan,"Home ITT":np.nan,"Vegas Score Away":np.nan,"Vegas Score Home":np.nan,"Vegas Connected":False,"Vegas Source":status})
    if not odds:
        return pd.DataFrame(rows)
    out=pd.DataFrame(rows)
    for ev in odds:
        home=team_name_match(ev.get("home_team","")); away=team_name_match(ev.get("away_team",""))
        if not home or not away: continue
        matchup=f"{away} @ {home}"
        if matchup not in set(out["Matchup"]): continue
        home_ml=away_ml=total=np.nan
        for book in ev.get("bookmakers",[])[:3]:
            for m in book.get("markets",[]):
                if m.get("key")=="h2h":
                    for o in m.get("outcomes",[]):
                        t=team_name_match(o.get("name",""));
                        if t==home and pd.isna(home_ml): home_ml=o.get("price",np.nan)
                        if t==away and pd.isna(away_ml): away_ml=o.get("price",np.nan)
                if m.get("key")=="totals" and pd.isna(total):
                    outs=m.get("outcomes",[])
                    if outs: total=outs[0].get("point",np.nan)
        hp=american_to_prob(home_ml); ap=american_to_prob(away_ml)
        home_itt=estimate_implied_runs(total,hp); away_itt=estimate_implied_runs(total,ap)
        home_score=np.clip(50 + (home_itt-4.2)*12 if pd.notna(home_itt) else 50,0,100)
        away_score=np.clip(50 + (away_itt-4.2)*12 if pd.notna(away_itt) else 50,0,100)
        idx=out.index[out["Matchup"].eq(matchup)]
        if len(idx):
            i=idx[0]
            out.loc[i,["Game Total","Away ML","Home ML","Away ITT","Home ITT","Vegas Score Away","Vegas Score Home","Vegas Connected","Vegas Source"]]=[total,away_ml,home_ml,away_itt,home_itt,away_score,home_score,True,"The Odds API current odds"]
    return out.round({"Game Total":1,"Away ITT":2,"Home ITT":2,"Vegas Score Away":1,"Vegas Score Home":1})

def bankroll_from_slate(score, risk_pref="Balanced"):
    try: score=float(score)
    except Exception: return 2.0
    if score>=90: pct=9.0
    elif score>=80: pct=6.5
    elif score>=70: pct=4.0
    elif score>=60: pct=2.5
    else: pct=1.5
    if risk_pref=="Conservative": pct*=0.65
    elif risk_pref=="Aggressive": pct*=1.25
    return round(float(np.clip(pct,1,10)),1)

def calculate_slate_rating(num_games, weather, stack_trends, scored, risk_pref="Balanced"):
    games_score=np.clip(45 + min(float(num_games or 0),15)*3.5, 35, 100)
    weather_score=85.0
    if weather is not None and not weather.empty and "Weather Risk" in weather.columns:
        avg_risk=pd.to_numeric(weather["Weather Risk"], errors="coerce").fillna(0).mean()
        high_games=(pd.to_numeric(weather["Weather Risk"], errors="coerce").fillna(0)>=55).sum()
        weather_score=np.clip(100-avg_risk-high_games*8,0,100)
    stack_depth=60.0
    if stack_trends is not None and not stack_trends.empty:
        st_scores=pd.to_numeric(stack_trends["Stack Trend Score"], errors="coerce").fillna(50)
        playable=(st_scores>=75).sum()
        elite=(st_scores>=90).sum()
        stack_depth=np.clip(45 + playable*6 + elite*5, 0, 100)
    portfolio_score=float(pd.to_numeric(scored.get("PBI",pd.Series([60])), errors="coerce").fillna(60).mean()) if scored is not None and not scored.empty else 60
    dup_score=70.0
    if scored is not None and "Dupes" in scored.columns:
        dup_score=float(dupe_score(scored["Dupes"]).mean())
    slate_score=np.clip(.25*games_score+.25*weather_score+.25*stack_depth+.15*portfolio_score+.10*dup_score,0,100)
    br=bankroll_from_slate(slate_score, risk_pref)
    if slate_score>=90: label="🟢 Attack slate"
    elif slate_score>=80: label="🟢 Strong slate"
    elif slate_score>=70: label="🟡 Normal slate"
    elif slate_score>=60: label="🟠 Caution slate"
    else: label="🔴 Thin / volatile slate"
    reasons=[f"{num_games} games", f"Weather score {weather_score:.1f}", f"Stack depth {stack_depth:.1f}", f"Portfolio PBI {portfolio_score:.1f}", f"Duplication health {dup_score:.1f}"]
    return {"Slate Rating":round(float(slate_score),1),"Slate Label":label,"Recommended BR %":br,"Reasons":"; ".join(reasons),"Games Score":round(float(games_score),1),"Weather Score":round(float(weather_score),1),"Stack Depth Score":round(float(stack_depth),1),"Portfolio Score":round(float(portfolio_score),1),"Duplication Score":round(float(dup_score),1)}

def make_stack_trends(scored, weather, park_df, team_df, scoring_df, master_df, vegas_df=None):
    teams=sorted([t for t in set(scored.get("Primary Stack Team",pd.Series(dtype=str)).dropna()) if str(t)!="None"])
    rows=[]
    team_scores=team_metric_scores(team_df); scoring_scores=team_metric_scores(scoring_df); master_scores=team_metric_scores(master_df,("Team",))
    park_scores=park_score_table(park_df)
    for t in teams:
        sub=scored[scored.get("Primary Stack Team").eq(t)]
        matchup=sub["Matchup"].mode().iloc[0] if "Matchup" in sub and not sub["Matchup"].mode().empty else "Unknown"
        ballpark=sub["Ballpark"].mode().iloc[0] if "Ballpark" in sub and not sub["Ballpark"].mode().empty else "Unknown"
        wrow=weather[weather["Matchup"].eq(matchup)] if not weather.empty else pd.DataFrame()
        wrisk=float(wrow["Weather Risk"].iloc[0]) if not wrow.empty else 0
        wboost=float(wrow["Hitter Weather Boost"].iloc[0]) if not wrow.empty else 0
        trend=np.nanmean([team_scores.get(t,50), scoring_scores.get(t,50), master_scores.get(t,50)])
        park=park_scores.get(ballpark,50)
        own=100-pct_rank(pd.Series([len(sub)]), True).iloc[0] if len(scored)>0 else 50
        avg_ckk=float(sub["CKK"].mean()) if "CKK" in sub else 50
        avg_fpe=float(sub["FPE"].mean()) if "FPE" in sub else 50
        leverage=float(sub["Lineup Edge"].mean()) if "Lineup Edge" in sub else 0
        lev_score=np.clip(50+leverage*5,0,100)
        vegas_score=np.nan; itt=np.nan; vegas_connected=False
        if vegas_df is not None and not vegas_df.empty and matchup in set(vegas_df["Matchup"]):
            vrow=vegas_df[vegas_df["Matchup"].eq(matchup)].iloc[0]
            vegas_connected=bool(vrow.get("Vegas Connected", False))
            if t==vrow.get("Home"):
                vegas_score=pd.to_numeric(vrow.get("Vegas Score Home",np.nan), errors="coerce"); itt=vrow.get("Home ITT",np.nan)
            elif t==vrow.get("Away"):
                vegas_score=pd.to_numeric(vrow.get("Vegas Score Away",np.nan), errors="coerce"); itt=vrow.get("Away ITT",np.nan)
        base_parts=[(.20,trend),(.15,park),(.15,(100-wrisk+wboost)),(.17,avg_fpe),(.13,avg_ckk),(.10,lev_score)]
        conf_parts=[(.25,trend),(.20,park),(.20,(100-wrisk)),(.10,avg_ckk),(.10,lev_score)]
        if vegas_connected and pd.notna(vegas_score):
            base_parts.append((.10,float(vegas_score)))
            conf_parts.append((.15,float(vegas_score)))
        # Renormalize if Vegas is not connected so missing odds do not silently become 50.
        stack_score=np.clip(sum(w*v for w,v in base_parts)/sum(w for w,_ in base_parts),0,100)
        confidence=np.clip(sum(w*v for w,v in conf_parts)/sum(w for w,_ in conf_parts),0,100)
        rec="Core/Heavy" if stack_score>=90 and confidence>=80 else "Playable" if stack_score>=75 else "Contrarian only" if stack_score>=65 else "Fade"
        rows.append({"Team":t,"Lineups":len(sub),"Matchup":matchup,"Ballpark":ballpark,"Overall Stack Grade":grade_from_score(stack_score),"Stack Trend Score":round(stack_score,1),"Stack Confidence":round(confidence,1),"Stack Recommendation":rec,"Vegas Score":round(float(vegas_score),1) if pd.notna(vegas_score) else np.nan,"Vegas Connected":vegas_connected,"Implied Team Total":round(float(itt),2) if pd.notna(itt) else np.nan,"Boom Score":round(.55*stack_score+.45*avg_fpe,1),"Bust Risk":round(100-stack_score+wrisk*.20,1),"Leverage Grade":grade_from_score(lev_score),"Weather Risk":round(wrisk,1),"Park Score":round(park,1),"Team Trend":round(trend,1),"Why Today":f"Trend {round(trend,1)}, park {round(park,1)}, weather risk {round(wrisk,1)}, Vegas {round(float(vegas_score),1) if pd.notna(vegas_score) else 'Not connected'}, confidence {round(confidence,1)}, leverage grade {grade_from_score(lev_score)}"})
    return pd.DataFrame(rows).sort_values("Stack Trend Score", ascending=False) if rows else pd.DataFrame()

def make_position_leverage(roo):
    p=normalize_player_df(roo)
    if p.empty: return pd.DataFrame()
    hitters=p[~p["Position"].str.contains("P", na=False)].copy()
    if hitters.empty: return hitters
    hitters["Projection Score"]=pct_rank(hitters["Median"], True)
    hitters["Ceiling Score"]=pct_rank(hitters["Ceiling"], True)
    hitters["Salary Score"]=pct_rank(hitters["Median"]/(hitters["Salary"].replace(0,np.nan)/1000), True)
    hitters["Ownership Leverage"]=(100-pct_rank(hitters["Ownership"], True)).clip(0,100)
    hitters["Trend Score"]=(.5*hitters["Projection Score"]+.5*hitters["Ceiling Score"]).round(1)
    hitters["Position Group"]=np.where(hitters["Position"].str.contains("OF",na=False),"OF",np.where(hitters["Position"].isin(["1B","2B","3B","SS","C"]),"IF","Other"))
    # IF/OF Priority: if the scoring sheet includes priority later, this can be merged; for now position-specific leverage is projection/ceiling/ownership/salary.
    hitters["Position Leverage Score"]=(.30*hitters["Ownership Leverage"]+.25*hitters["Ceiling Score"]+.20*hitters["Projection Score"]+.15*hitters["Salary Score"]+.10*hitters["Trend Score"]).round(1).clip(0,100)
    hitters["Leverage Tag"]=pd.cut(hitters["Position Leverage Score"], bins=[-1,45,65,80,101], labels=["Chalk Fade","Neutral","Good Leverage","Elite Leverage"])
    return hitters.sort_values("Position Leverage Score", ascending=False)

def apply_trend_adjustments(scored, stack_trends):
    out=scored.copy()
    if stack_trends is None or stack_trends.empty or "Primary Stack Team" not in out.columns:
        out["CKK Trend Adj"]=out.get("MLB CKK",out["CKK"])
        out["FPE Trend Adj"]=out["FPE"]
        return out
    mp=dict(zip(stack_trends["Team"], stack_trends["Stack Trend Score"]))
    stack_score=out["Primary Stack Team"].map(mp).fillna(50)
    weather_score=out["Weather Score"] if "Weather Score" in out.columns else 50
    adjustment=((stack_score-50)*.08 + (pd.to_numeric(weather_score,errors="coerce").fillna(50)-50)*.04).clip(-8,8)
    out["Trend Adjustment"] = adjustment.round(2)
    out["CKK Trend Adj"]=(out.get("MLB CKK",out["CKK"])+adjustment).round(2).clip(0,100)
    out["FPE Trend Adj"]=(out["FPE"]+adjustment*.75).round(2).clip(0,100)
    return out

def slate_story(stack_trends, pitcher_trends, hitter_trends, weather):
    parts=[]
    if stack_trends is not None and not stack_trends.empty:
        top=stack_trends.iloc[0]
        parts.append(f"The slate currently favors {top['Team']} stacks: Stack Trend {top['Stack Trend Score']} ({top['Overall Stack Grade']}), matchup {top['Matchup']}, park {top['Park Score']}, weather risk {top['Weather Risk']}.")
    if pitcher_trends is not None and not pitcher_trends.empty:
        top=pit= pitcher_trends.iloc[0]
        parts.append(f"Top pitcher trend: {top['Player']} ({top.get('Team','')}) with a {top['Pitcher Trend Grade']} grade, ceiling meter {top['Ceiling Meter']}, bust probability {top['Bust Probability']}.")
    if hitter_trends is not None and not hitter_trends.empty:
        h=hitter_trends.iloc[0]
        parts.append(f"Top hitter trend/value: {h['Player']} ({h.get('Team','')}) with Hitter Trend {h['Hitter Trend Score']} and Boom Probability {h['Boom Probability']}.")
    if weather is not None and not weather.empty:
        risky=weather.sort_values("Weather Risk", ascending=False).head(1).iloc[0]
        if risky["Weather Risk"]>=25:
            parts.append(f"Weather watch: {risky['Matchup']} at {risky['Ballpark']} shows {risky['Risk Label']} risk ({risky['Weather Risk']}).")
        else:
            parts.append("Weather looks mostly manageable right now based on the ballpark API risk table.")
    if not parts: return "Upload MLB files to generate a slate story."
    return " ".join(parts)


st.title("🏆 CKK Portfolio Builder v13.2")
st.caption("Smart Import Engine • CKK/FPE/Confidence • MLB Trends • Vegas Impact • Slate Rating + Bankroll Coach")

with st.sidebar:
    st.header("Settings")
    sport_manual=st.selectbox("Sport override", ["Auto","MLB","NBA","MMA","Other"], index=0)
    auto_weather=st.toggle("Auto-fetch ballpark weather via Open-Meteo API", value=True, help="Uses stadium coordinates from the DK Salaries home team. Roof parks get reduced weather risk.")
    st.divider()
    st.subheader("Slate + Bankroll")
    manual_num_games=st.number_input("Number of games on slate", min_value=1, max_value=20, value=8, step=1)
    bankroll=st.number_input("Bankroll $", min_value=0.0, value=1000.0, step=50.0)
    risk_pref=st.selectbox("Risk preference", ["Conservative","Balanced","Aggressive"], index=1)
    st.divider()
    st.subheader("Vegas API")
    odds_api_key=st.text_input("The Odds API key optional", type="password", help="Optional. If blank, Vegas impact stays neutral at 50. Current build uses live current totals/moneylines; opening line movement can be added later if your odds source provides it.")
    st.caption("Tip: upload all slate CSVs at once. Weather is pulled live from an API, not uploaded.")

uploads=st.file_uploader("📁 Drag/drop all slate CSVs here", type=["csv"], accept_multiple_files=True)
classified: Dict[str, Tuple[str,pd.DataFrame]]={}
unknown=[]
if uploads:
    for up in uploads:
        try:
            df=clean_cols(read_csv(up)); kind=classify_file(up.name, df)
            if kind=="Unknown": unknown.append((up.name,df))
            else: classified[kind]=(up.name,df)
        except Exception as e:
            st.error(f"Could not read {up.name}: {e}")

st.subheader("Smart Import Status")
needed_base=["Portfolio Manager"]
needed_mlb=["DK Salaries","MLB Matchups Master","Park Factors","Team Gamelogs","Scoring %"]
cols=st.columns(4)
all_types=["Portfolio Manager","ROO Projections","DK Salaries","MLB Matchups Master","Park Factors","Team Gamelogs","Scoring %"]
for n,t in enumerate(all_types):
    with cols[n%4]:
        if t in classified:
            st.success(f"✅ {t}\n\n{classified[t][0]}")
        else:
            st.info(f"⬜ {t}")
if unknown:
    with st.expander("Unknown files"):
        for name,df in unknown: st.write(name, list(df.columns)[:12])

pm=classified.get("Portfolio Manager",(None,None))[1]
roo=classified.get("ROO Projections",(None,None))[1]
dk=classified.get("DK Salaries",(None,None))[1]
park=classified.get("Park Factors",(None,None))[1]
team=classified.get("Team Gamelogs",(None,None))[1]
scoring=classified.get("Scoring %",(None,None))[1]
master=classified.get("MLB Matchups Master",(None,None))[1]

sport="Unknown"
if pm is not None:
    c=set(str(x).upper() for x in pm.columns)
    sport="MLB" if len(c & {"SP1","SP2","P","C","1B","2B","3B","SS","OF1","OF2","OF3"})>=5 else "NBA" if len(c & {"PG","SG","SF","PF","C","G","F","UTIL"})>=5 else "Unknown"
if sport_manual != "Auto": sport=sport_manual
st.write(f"Detected sport: **{sport}**")

if pm is None:
    st.warning("Upload a Portfolio Manager export to unlock analysis.")
    st.stop()

scored=add_scores(pm)

if sport=="MLB":
    game_map=build_game_map(dk)
    weather=summarize_weather(game_map, None, auto_weather)
    vegas=build_vegas_table(game_map, odds_api_key)
    scored=add_mlb_context(scored, game_map, weather, park, team, scoring, master)

    st.subheader("🌦️ MLB Weather Risk by Matchup and Teams")
    if weather.empty:
        st.warning("Upload DK Salaries with Game Info to map teams → home ballpark → live weather API risk.")
    else:
        st.dataframe(weather, use_container_width=True, hide_index=True)

    st.subheader("🏟️ DK Salaries Game → Ballpark Map")
    if not game_map.empty: st.dataframe(game_map[["Matchup","Away","Home","Ballpark","City","Roof/Potential Roof"]], use_container_width=True, hide_index=True)

    st.subheader("🎰 Vegas Impact by Matchup")
    if vegas.empty:
        st.info("Upload DK Salaries to map matchups. Add a The Odds API key in the sidebar for live totals/moneylines.")
    else:
        if "Vegas Connected" in vegas.columns and not vegas["Vegas Connected"].fillna(False).any():
            st.warning("Vegas is NOT connected. Stack ratings are excluding Vegas from the formula instead of using fake neutral 50s. Add/check your Odds API key to enable live odds.")
        else:
            st.success("Vegas connected. Live totals/moneylines are being used in Stack Score and Stack Confidence.")
        st.dataframe(vegas, use_container_width=True, hide_index=True)

    pitcher_trends=make_pitcher_trends(roo, game_map, weather, team, scoring, master)
    hitter_trends=make_hitter_trends(roo, game_map, weather, team, scoring, master)
    stack_trends=make_stack_trends(scored, weather, park, team, scoring, master, vegas)
    position_leverage=make_position_leverage(roo)
    scored=apply_trend_adjustments(scored, stack_trends)
else:
    pitcher_trends=pd.DataFrame(); hitter_trends=pd.DataFrame(); stack_trends=pd.DataFrame(); position_leverage=pd.DataFrame(); weather=pd.DataFrame(); vegas=pd.DataFrame()

st.subheader("🏆 Portfolio Summary")
metrics=st.columns(6)
for col,label in zip(metrics,["CKK","FPE","Confidence","PBI","Print Probability","MLB Stack Score" if sport=="MLB" else "CKK"]):
    if label in scored.columns: col.metric(label, f"{scored[label].mean():.1f}")

if sport=="MLB":
    slate=calculate_slate_rating(manual_num_games, weather, stack_trends, scored, risk_pref)
    st.subheader("🎯 Slate Rating + Bankroll Coach")
    scols=st.columns(5)
    scols[0].metric("Slate Rating", f"{slate['Slate Rating']}/100")
    scols[1].metric("Slate Type", slate["Slate Label"])
    scols[2].metric("Recommended BR", f"{slate['Recommended BR %']}%")
    scols[3].metric("Dollar Amount", f"${bankroll*slate['Recommended BR %']/100:,.0f}")
    scols[4].metric("Games", manual_num_games)
    st.caption(slate["Reasons"])

    tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs(["📈 Stack Trend Engine","🔥 Hitter Trend Engine","⚾ Pitcher Trend Engine","🧬 IF/OF Leverage","🧠 Slate Story","📡 Live Stack Tracker","🎯 Slate Coach"] )
    with tab1:
        st.caption("Combines team recent offense, scoring/master sheet, park factor, weather, portfolio CKK/FPE, and leverage.")
        if stack_trends.empty: st.info("Upload MLB portfolio + DK salaries + any trend/scoring/master sheets to populate stack trends.")
        else: st.dataframe(stack_trends, use_container_width=True, hide_index=True)
    with tab2:
        st.caption("Uses ROO projection/ceiling/value/ownership plus team trend and weather context. Advanced Statcast/pitch-type data can be added when available.")
        if hitter_trends.empty: st.info("Upload ROO projections to populate hitter trends.")
        else: st.dataframe(hitter_trends[[c for c in ["Player","Position","Team","Salary","Median","Ceiling","Ownership","Hitter Trend Score","Hitter Trend Grade","Ceiling Rating","Value Rating","Boom Probability","Why Today"] if c in hitter_trends.columns]].head(80), use_container_width=True, hide_index=True)
    with tab3:
        st.caption("Uses ROO pitcher projection/ceiling/value/ownership. Last 3/5/10 starts, K/BB, velocity, umpire, pitch-type effectiveness can plug in next when source sheets/APIs are added.")
        if pitcher_trends.empty: st.info("Upload ROO projections with pitchers to populate pitcher trends.")
        else: st.dataframe(pitcher_trends[[c for c in ["Player","Team","Salary","Median","Ceiling","Ownership","Pitcher Trend Score","Pitcher Trend Grade","Hot/Neutral/Cold","Risk Meter","Ceiling Meter","Bust Probability","Why Today"] if c in pitcher_trends.columns]], use_container_width=True, hide_index=True)
    with tab4:
        st.caption("Separate IF/OF leverage using ownership, projection, ceiling, salary value, and trend score.")
        if position_leverage.empty: st.info("Upload ROO projections to populate IF/OF leverage.")
        else:
            group=st.selectbox("Position group", ["All","IF","OF"], key="poslevgroup")
            view=position_leverage if group=="All" else position_leverage[position_leverage["Position Group"].eq(group)]
            st.dataframe(view[[c for c in ["Player","Position","Position Group","Team","Salary","Median","Ceiling","Ownership","Position Leverage Score","Leverage Tag"] if c in view.columns]].head(100), use_container_width=True, hide_index=True)
    with tab5:
        st.markdown("### Why Today? / Slate Story")
        st.write(slate_story(stack_trends, pitcher_trends, hitter_trends, weather))
        if not stack_trends.empty:
            st.write("**Top stack explanations**")
            st.dataframe(stack_trends[["Team","Overall Stack Grade","Stack Trend Score","Boom Score","Bust Risk","Leverage Grade","Why Today"]].head(10), use_container_width=True, hide_index=True)
    with tab6:
        st.caption("Framework for after-lock live tracking. Current build shows stack exposure/contribution structure; live DK scoring feed can be plugged in later.")
        if "Primary Stack Team" in scored.columns:
            live=scored.groupby("Primary Stack Team", dropna=False).agg(Lineups=("Primary Stack Team","size"), Avg_CKK=("CKK","mean"), Avg_FPE=("FPE","mean"), Avg_Trend_CKK=("CKK Trend Adj" if "CKK Trend Adj" in scored.columns else "CKK","mean")).reset_index()
            st.dataframe(live, use_container_width=True, hide_index=True)
    with tab7:
        st.markdown("### Slate Coach")
        st.write(f"**{slate['Slate Label']}** — Recommended bankroll exposure: **{slate['Recommended BR %']}%** (${bankroll*slate['Recommended BR %']/100:,.0f}).")
        detail=pd.DataFrame([{k:v for k,v in slate.items() if k not in ['Reasons']}])
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.markdown("**Why:** " + slate['Reasons'])
        if not stack_trends.empty:
            st.markdown("**Playable stack thresholds:** A-/A/A+ are core candidates, B/B+ are playable, B- is mini/secondary, C+ is contrarian only.")
            st.dataframe(stack_trends[[c for c in ["Team","Overall Stack Grade","Stack Trend Score","Stack Confidence","Stack Recommendation","Vegas Connected","Vegas Score","Implied Team Total","Weather Risk","Leverage Grade","Why Today"] if c in stack_trends.columns]].head(20), use_container_width=True, hide_index=True)

st.subheader("📊 Lineup Scores")
show_cols=[c for c in ["Decision","CKK","FPE","Confidence","PBI","Print Probability","MLB CKK","CKK Trend Adj","FPE Trend Adj","Trend Adjustment","MLB Stack Score","Primary Stack Team","Matchup","Ballpark","Home/Away","Weather Label","Weather Risk","Park Score","Team/Master Trend Score","Win%","Finish_percentile","Lineup Edge","Geomean","Diversity","Weighted Own","Dupes","median","Stack"] if c in scored.columns]
st.dataframe(scored[show_cols].sort_values("PBI", ascending=False), use_container_width=True, hide_index=True)

st.subheader("🤖 What Would CKK Do?")
weak=scored.sort_values("PBI").head(min(10,len(scored)))
strong=scored.sort_values("PBI", ascending=False).head(min(10,len(scored)))
c1,c2=st.columns(2)
with c1:
    st.write("**Top keep/build-around lineups**")
    st.dataframe(strong[[c for c in ["PBI","CKK","FPE","Confidence","Decision","Primary Stack Team","Matchup","Weather Label"] if c in strong.columns]], hide_index=True, use_container_width=True)
with c2:
    st.write("**First lineups to review/remove**")
    st.dataframe(weak[[c for c in ["PBI","CKK","FPE","Confidence","Decision","Primary Stack Team","Matchup","Weather Label"] if c in weak.columns]], hide_index=True, use_container_width=True)

csv=scored.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download scored portfolio CSV", csv, file_name="ckk_scored_portfolio_v13.csv", mime="text/csv")
