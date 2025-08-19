import io
import re
import math
import base64
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

TMD_URL = "https://earthquake.tmd.go.th/inside.html"

# -------------------- Utilities --------------------
def _clean_num(s: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", s))

def _parse_latlon(lat_s: str, lon_s: str):
    lat = _clean_num(lat_s)
    lon = _clean_num(lon_s)
    s_lat = lat_s.strip().upper()
    s_lon = lon_s.strip().upper()
    if s_lat.endswith(("S", "ใต้")):
        lat = -lat
    if s_lon.endswith(("W", "ตะวันตก")):
        lon = -lon
    return lat, lon

def _looks_region_th(s: str) -> bool:
    return any(k in s for k in ("ประเทศไทย", "จ.", "อ.", "ต.", "ตำบล", "อำเภอ", "จังหวัด"))

def _parse_datetime_th_block(block_before_mag: list[str]) -> tuple[str, str]:
    re_dt = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
    time_th, time_utc = "", ""
    for s in block_before_mag[-6:][::-1]:
        if re_dt.search(s):
            if "UTC" in s.upper():
                m = re_dt.search(s)
                if m:
                    time_utc = m.group(0)
            else:
                m = re_dt.search(s)
                if m and not time_th:
                    time_th = m.group(0)
        if time_th and time_utc:
            break
    return time_th, time_utc

def reverse_geocode_th(lat: float, lon: float, timeout=15):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": str(lat),
        "lon": str(lon),
        "accept-language": "th",
        "zoom": 12,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "Mozilla/5.0 (Python requests for quake QA)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        addr = js.get("address", {}) or {}
    except Exception:
        addr = {}

    tambon = (
        addr.get("subdistrict") or
        addr.get("town") or
        addr.get("village") or
        addr.get("suburb") or
        ""
    )
    amphoe = (addr.get("district") or addr.get("county") or "")
    changwat = (addr.get("province") or addr.get("state") or "")
    return {"tambon": tambon, "amphoe": amphoe, "changwat": changwat}

def _parse_tambon_from_text(s: str):
    tambon = ""
    m_a = re.search(r"(อ\.|อำเภอ)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    m_c = re.search(r"(จ\.|จังหวัด)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    amphoe = (m_a.group(2).strip() if m_a else "")
    changwat = (m_c.group(2).strip() if m_c else "")
    m_t = re.search(r"(ต\.|ตำบล)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    tambon = (m_t.group(2).strip() if m_t else "")
    return tambon, amphoe, changwat

def fetch_latest_event_in_thailand():
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(TMD_URL, timeout=20, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]

    events = []
    i = 0
    re_mag = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
    re_deg = re.compile(r"^[\-]?[0-9]+(?:\.[0-9]+)?\s*°\s*[NSEW]?$", re.IGNORECASE)
    re_num = re.compile(r"^[\-]?[0-9]+(?:\.[0-9]+)?$")

    while i < len(lines) - 7:
        s_mag = lines[i]
        if re_mag.fullmatch(s_mag):
            lat_s = lines[i+1] if i+1 < len(lines) else ""
            lon_s = lines[i+2] if i+2 < len(lines) else ""
            dep_s = lines[i+3] if i+3 < len(lines) else ""
            cand4 = lines[i+4] if i+4 < len(lines) else ""
            cand5 = lines[i+5] if i+5 < len(lines) else ""
            region_s = cand5 if re_num.fullmatch(cand4 or "") else cand4
            pre_block = lines[max(0, i-6):i]
            time_th, time_utc = _parse_datetime_th_block(pre_block)
            dep_num_s = re.sub(r"[^0-9.\-]", "", dep_s) or "0"
            if re_deg.fullmatch(lat_s) and re_deg.fullmatch(lon_s) and re_num.fullmatch(dep_num_s):
                try:
                    mag = float(s_mag)
                    lat, lon = _parse_latlon(lat_s, lon_s)
                    depth = float(dep_num_s)
                    events.append(dict(
                        mag=mag, lat=lat, lon=lon, depth=depth, region=region_s,
                        time_th=time_th, time_utc=time_utc
                    ))
                    i += 6
                    continue
                except Exception:
                    pass
        i += 1

    if not events:
        return None
    for ev in events:
        if _looks_region_th(ev["region"]):
            return ev
    return events[0]

def _fmt_th_datetime(s: str) -> str:
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return s

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _fmt_num(x, digits=3):
    """แปลงตัวเลขเป็นสตริง ตัดศูนย์/จุดท้าย ๆ เพื่อดู ‘เหมือน API’ มากขึ้น"""
    s = f"{float(x):.{digits}f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"

def compute_overlay_from_event(ev: dict):
    """
    รับ ev = {lat, lon, mag, depth, region, time_th, time_utc}
    คืนผล JSON-ready:
    {
      "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
      "epicenter": [lat, lon],
      "image_data_url": "data:image/png;base64,...",
      "popup_html": "...",
      "meta": {...},
      "pga_points": [...]
    }
    """
    lat = float(ev["lat"]); lon = float(ev["lon"]); mag = float(ev["mag"]); depth = float(ev["depth"])
    region_text = ev.get("region", "")
    time_th = ev.get("time_th", "")
    time_utc = ev.get("time_utc", "")

    # reverse geocode
    geo = reverse_geocode_th(lat, lon)
    tambon, amphoe, changwat = geo["tambon"], geo["amphoe"], geo["changwat"]
    if not (amphoe or changwat):
        t2, a2, c2 = _parse_tambon_from_text(region_text)
        tambon = tambon or t2
        amphoe = amphoe or a2
        changwat = changwat or c2

    # -------------------- พารามิเตอร์กริดและ CY08 --------------------
    half_box_deg = 2.0
    spacing = 0.05

    latmin = lat - half_box_deg
    latmax = lat + half_box_deg
    lonmin = lon - half_box_deg
    lonmax = lon + half_box_deg

    n_lat = int(round((latmax - latmin) / spacing))
    n_lon = int(round((lonmax - lonmin) / spacing))

    lat_vals = latmin + spacing * np.arange(n_lat)
    lon_vals = lonmin + spacing * np.arange(n_lon)

    LAT, LON = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    # กลไก (เหมือน VB)
    strike = 0.0
    dip    = 45.0
    rake   = 0.0

    a_km2 = 10 ** ((mag - 4.07) / 0.98)
    w_km  = math.sqrt(a_km2 / 2.0)
    l_km  = 2.0 * w_km
    ztor  = depth - (w_km * math.sin(dip * math.pi/180.0) / 2.0)
    ztor  = max(0.0, ztor)

    c2 = 1.06; c3 = 3.45; c4 = -2.1; c4a = -0.5; crb = 50.0; chm = 3.0; cy3 = 4.0
    c1 = -1.2687; c1a = 0.1; c1b = -0.255; cn = 2.996; cm = 4.184; c5 = 6.16
    c6 = 0.4893; c7 = 0.0512; c7a = 0.086; c9 = 0.79; c9a = 1.5005
    c10 = -0.3218; cy1 = -0.00804; cy2 = -0.00785

    Frv = 1.0 if (30 <= rake <= 150) else 0.0
    Fnm = 1.0 if (-120 <= rake <= -60) else 0.0
    Ass = 0.0
    ff1 = c1 + (c1a * Frv + c1b * Fnm + c7 * (ztor - 4.0)) * (1.0 - Ass) + (c10 + c7a * (ztor - 4.0)) * Ass

    max1 = max(mag - chm, 0.0)
    xx = c6 * max1
    coshxx = (np.exp(xx) + np.exp(-xx)) / 2.0
    ff2 = c2 * (mag - 6.0) + ((c2 - c3) / cn) * np.log1p(np.exp(cn * (cm - mag)))

    DEG2RAD = math.pi / 180.0
    coslat = np.cos(LAT * DEG2RAD)
    dx_km = (lon - LON) * coslat * 111.0
    dy_km = (lat - LAT) * 111.0
    dist_km = np.sqrt(dx_km**2 + dy_km**2)

    ff33 = c4 * np.log(dist_km + c5 * coshxx)
    ff43 = (c4a - c4) * np.log(np.sqrt(dist_km**2 + crb**2))

    max2 = max(mag - cy3, 0.0)
    xxx = max2
    coshxxx = (np.exp(xxx) + np.exp(-xxx)) / 2.0
    ff53 = (cy1 + cy2 / (coshxxx)) * dist_km

    lnPGA_g = ff1 + ff2 + ff33 + ff43 + ff53
    PGA = np.exp(lnPGA_g) * 100.0

    # Soft mask
    sigma_km = 50.0
    dist_ep = _haversine(lat, lon, LAT, LON)
    soft_mask = np.exp(-0.5 * (dist_ep / sigma_km)**2)
    masked_pga = PGA * soft_mask
    masked_pga = np.where(np.isnan(masked_pga), 0.0, masked_pga)
    masked_pga = np.where(masked_pga < 0, 0.0, masked_pga)

    pga_max = float(np.max(masked_pga))

    # --- PGA grid points for nearest lookup ---
    pga_points = []
    for i in range(LAT.shape[0]):
        for j in range(LAT.shape[1]):
            pga_points.append({
                "lat": float(LAT[i, j]),
                "lon": float(LON[i, j]),
                "pga": float(masked_pga[i, j])
            })

    # วาดภาพโปร่งใสเป็น PNG
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    norm = Normalize(vmin=float(masked_pga.min()), vmax=float(masked_pga.max()))
    ax.imshow(
        masked_pga,
        extent=[lonmin, lonmax, latmin, latmax],
        origin="lower",
        cmap="jet",
        norm=norm,
        alpha=0.55,
        interpolation="bicubic"
    )
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    time_th_fmt = _fmt_th_datetime(time_th) if time_th else "-"

    # ---------- Popup (เพิ่มแสดงขนาดแผ่นดินไหวจาก API) ----------
    popup_html = f"""
<div style="line-height:1.5; font-size:1.2em; color:red; padding:4px; text-align:center;">
  <strong>แผ่นดินไหว</strong><br>
</div>
<div style="line-height:1.2; font-size:1.05em">
  วันเวลา: <b>{time_th_fmt} น.</b><br>
  ขนาด: <b>{_fmt_num(mag, 2)}</b><br>
  จุดศูนย์กลาง: <b>{region_text}</b><br>
  ค่าระดับการสั่นสะเทือนสูงสุด: <b><span style="color:red;">{_fmt_num(pga_max, 3)}</span> %g</b><br>
</div>
""".strip()

    # ---------- Metadata (เก็บค่าจาก API แบบไม่ปัด) ----------
    meta = {
        "time_th": time_th_fmt,
        "time_utc": time_utc or "-",
        "region_text": region_text,
        "tambon": tambon or "-",
        "amphoe": amphoe or "-",
        "changwat": changwat or "-",
        "lat": float(round(lat, 1)),     # สำหรับแสดงสวย ๆ
        "lon": float(round(lon, 1)),
        "mag_api": float(mag),           # ตาม API (ไม่ปัดทศนิยม)
        "depth_km": float(depth),
        "pga_max": float(round(pga_max, 2))
    }

    return {
        "bounds": [[float(latmin), float(lonmin)], [float(latmax), float(lonmax)]],
        "epicenter": [float(lat), float(lon)],
        "image_data_url": data_url,
        "popup_html": popup_html,
        "meta": meta,
        "pga_points": pga_points
    }

def run_pipeline():
    """
    ดึงเหตุการณ์ล่าสุดในไทย/ใกล้ไทยจาก TMD -> คำนวณ PGA -> ทำภาพ overlay และข้อมูลสำหรับเว็บ
    """
    ev = fetch_latest_event_in_thailand()
    if not ev:
        raise RuntimeError("ไม่พบเหตุการณ์จาก TMD หรือโครงสร้างหน้าเว็บเปลี่ยนไป")
    result = compute_overlay_from_event(ev)
    return result
