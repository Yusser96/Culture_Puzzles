def load_registry(cfg):
    return cfg.get("lang_regions", [])

def region_factors(key, cfg):
    rec = next((r for r in load_registry(cfg) if r["key"] == key), {})
    region = key.split("_", 1)[1] if "_" in key else ""
    return {"base_language": rec.get("base_language", "UNKNOWN"),
            "region": region, "language_region": key,
            "wiki": rec.get("wiki"), "flores": rec.get("flores"), "opus": rec.get("opus")}
