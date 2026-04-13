from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.demo import TRIAGE_LABELS, transform_sample
from src.models import get_models
from src.preprocessing import PreprocessConfig, load_and_preprocess
from src.train import train_models

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "triage.csv"

DEFAULT_FORM_VALUES = {
    "Yas": "67",
    "Nabiz": "112",
    "Ates": "38.4",
    "Sistolik_Tansiyon": "95",
    "Oksijen_Saturasyonu": "91",
    "Solunum_Sayisi": "24",
    "Sikayet": "NefesDarligi",
}

FIELD_CONFIG = {
    "Yas": {"label": "Yaş", "min": 0, "max": 120, "hint": "0 - 120"},
    "Nabiz": {"label": "Nabız", "min": 20, "max": 220, "hint": "20 - 220 bpm"},
    "Ates": {"label": "Ateş", "min": 30, "max": 45, "hint": "30.0 - 45.0 °C"},
    "Sistolik_Tansiyon": {"label": "Sistolik Tansiyon", "min": 50, "max": 250, "hint": "50 - 250 mmHg"},
    "Oksijen_Saturasyonu": {"label": "Oksijen Satürasyonu", "min": 50, "max": 100, "hint": "50 - 100 %"},
    "Solunum_Sayisi": {"label": "Solunum Sayısı", "min": 5, "max": 80, "hint": "5 - 80 /dk"},
}

TRIAGE_COLORS = {
    2: "#b91c1c",
    3: "#c2410c",
    4: "#0369a1",
    5: "#15803d",
}


class TriageApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Emergency Triage Prediction")
        self.root.geometry("860x640")
        self.root.minsize(860, 640)
        self.root.configure(bg="#0f172a")

        self.model = None
        self.metadata = None
        self.available_complaints: list[str] = []
        self.entries: dict[str, ttk.Entry] = {}
        self.value_vars: dict[str, tk.StringVar] = {}
        self.complaint_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Model yükleniyor...")
        self.result_var = tk.StringVar(value="Henüz tahmin yapılmadı.")
        self.result_detail_var = tk.StringVar(value="Hasta verilerini girip 'Tahmin Et' butonuna bas.")
        self.model_info_var = tk.StringVar(value="MLP modeli hazırlanıyor...")

        self._configure_styles()
        self._build_ui()
        self._load_model()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111827")
        style.configure("Panel.TLabelframe", background="#111827", foreground="#f8fafc")
        style.configure("Panel.TLabelframe.Label", background="#111827", foreground="#f8fafc", font=("Arial", 12, "bold"))
        style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc", font=("Arial", 22, "bold"))
        style.configure("Subtitle.TLabel", background="#0f172a", foreground="#cbd5e1", font=("Arial", 10))
        style.configure("Body.TLabel", background="#111827", foreground="#e5e7eb", font=("Arial", 10))
        style.configure("Hint.TLabel", background="#111827", foreground="#94a3b8", font=("Arial", 9))
        style.configure("Result.TLabel", background="#111827", foreground="#f8fafc", font=("Arial", 20, "bold"))
        style.configure("Secondary.TLabel", background="#111827", foreground="#cbd5e1", font=("Arial", 10))
        style.configure("App.TEntry", fieldbackground="#0b1220", foreground="#f8fafc", insertcolor="#f8fafc", padding=8)
        style.map("App.TEntry", fieldbackground=[("focus", "#111c33")])
        style.configure("App.TCombobox", fieldbackground="#0b1220", background="#0b1220", foreground="#f8fafc", padding=8)
        style.configure("Primary.TButton", font=("Arial", 10, "bold"), padding=(14, 10))
        style.configure("Secondary.TButton", font=("Arial", 10), padding=(14, 10))

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        ttk.Label(header, text="Emergency Triage Prediction System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Vital bulgular ve şikayet bilgisini kullanarak MLP modeli ile triyaj seviyesi tahmini yapar.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        left = ttk.LabelFrame(container, text="Hasta Bilgileri", style="Panel.TLabelframe", padding=18)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(1, weight=1)

        row_idx = 0
        for field_key, cfg in FIELD_CONFIG.items():
            ttk.Label(left, text=cfg["label"], style="Body.TLabel").grid(row=row_idx, column=0, sticky="w", pady=(0, 2))
            var = tk.StringVar(value=DEFAULT_FORM_VALUES[field_key])
            entry = ttk.Entry(left, textvariable=var, style="App.TEntry", width=28)
            entry.grid(row=row_idx, column=1, sticky="ew", padx=(12, 0), pady=(0, 2))
            ttk.Label(left, text=cfg["hint"], style="Hint.TLabel").grid(row=row_idx + 1, column=1, sticky="w", padx=(12, 0), pady=(0, 10))
            self.entries[field_key] = entry
            self.value_vars[field_key] = var
            row_idx += 2

        ttk.Label(left, text="Şikayet", style="Body.TLabel").grid(row=row_idx, column=0, sticky="w", pady=(0, 2))
        self.complaint_dropdown = ttk.Combobox(left, textvariable=self.complaint_var, state="readonly", style="App.TCombobox")
        self.complaint_dropdown.grid(row=row_idx, column=1, sticky="ew", padx=(12, 0), pady=(0, 2))
        ttk.Label(left, text="Mevcut şikayet kategorilerinden birini seç.", style="Hint.TLabel").grid(
            row=row_idx + 1, column=1, sticky="w", padx=(12, 0), pady=(0, 12)
        )

        button_bar = ttk.Frame(left, style="Card.TFrame")
        button_bar.grid(row=row_idx + 2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(button_bar, text="Tahmin Et", command=self.predict, style="Primary.TButton").pack(side="left")
        ttk.Button(button_bar, text="Demo Veriyi Doldur", command=self.fill_demo_values, style="Secondary.TButton").pack(side="left", padx=10)
        ttk.Button(button_bar, text="Temizle", command=self.clear_form, style="Secondary.TButton").pack(side="left")

        right = ttk.Frame(container, style="App.TFrame")
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)

        summary_card = ttk.LabelFrame(right, text="Model Bilgisi", style="Panel.TLabelframe", padding=18)
        summary_card.grid(row=0, column=0, sticky="ew")
        ttk.Label(summary_card, textvariable=self.model_info_var, style="Body.TLabel", wraplength=290, justify="left").pack(anchor="w")

        result_card = ttk.LabelFrame(right, text="Tahmin Sonucu", style="Panel.TLabelframe", padding=18)
        result_card.grid(row=1, column=0, sticky="ew", pady=12)
        self.result_badge = tk.Label(
            result_card,
            textvariable=self.result_var,
            bg="#111827",
            fg="#f8fafc",
            font=("Arial", 22, "bold"),
            anchor="w",
            justify="left",
        )
        self.result_badge.pack(fill="x")
        ttk.Label(result_card, textvariable=self.result_detail_var, style="Secondary.TLabel", wraplength=300, justify="left").pack(anchor="w", pady=(10, 0))

        status_card = ttk.LabelFrame(right, text="Durum", style="Panel.TLabelframe", padding=18)
        status_card.grid(row=2, column=0, sticky="nsew")
        ttk.Label(status_card, textvariable=self.status_var, style="Body.TLabel", wraplength=300, justify="left").pack(anchor="w")
        ttk.Label(
            status_card,
            text="İpucu: Sunum sırasında önce 'Demo Veriyi Doldur', sonra 'Tahmin Et' butonuna basabilirsin.",
            style="Hint.TLabel",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _load_model(self) -> None:
        try:
            config = PreprocessConfig(
                test_size=0.2,
                random_state=42,
                encoding="onehot",
                scaling="zscore",
                clip_z_threshold=0.0,
            )
            X_train, _X_test, y_train, _y_test, metadata = load_and_preprocess(str(DATA_PATH), config)
            models = get_models()
            trained = train_models(models, X_train, y_train)
            self.model = trained["MLP"]
            self.metadata = metadata
            self.available_complaints = metadata.get("categories", []) or ["NefesDarligi"]
            self.complaint_dropdown["values"] = self.available_complaints
            default_complaint = DEFAULT_FORM_VALUES["Sikayet"]
            self.complaint_var.set(default_complaint if default_complaint in self.available_complaints else self.available_complaints[0])
            self.model_info_var.set(
                f"Aktif model: MLP\nÖrnek sayısı: {metadata['sample_count']}\nÖzellik sayısı: {len(metadata['feature_names'])}\n"
                f"Sınıf dağılımı: {metadata['class_distribution']}"
            )
            self.status_var.set("Model başarıyla yüklendi. Uygulama tahmine hazır.")
        except Exception as exc:
            self.status_var.set(f"Model yüklenemedi: {exc}")
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu:\n{exc}")

    def fill_demo_values(self) -> None:
        for field_key, value in DEFAULT_FORM_VALUES.items():
            if field_key == "Sikayet":
                continue
            self.value_vars[field_key].set(value)
        complaint = DEFAULT_FORM_VALUES["Sikayet"]
        if complaint in self.available_complaints:
            self.complaint_var.set(complaint)
        elif self.available_complaints:
            self.complaint_var.set(self.available_complaints[0])
        self.result_var.set("Hazır")
        self.result_detail_var.set("Demo hasta verisi dolduruldu. Şimdi tahmin alabilirsin.")
        self.result_badge.config(fg="#f8fafc")

    def clear_form(self) -> None:
        for var in self.value_vars.values():
            var.set("")
        if self.available_complaints:
            self.complaint_var.set(self.available_complaints[0])
        else:
            self.complaint_var.set("")
        self.result_var.set("Henüz tahmin yok")
        self.result_detail_var.set("Tüm alanları doldurup tekrar deneyebilirsin.")
        self.result_badge.config(fg="#f8fafc")
        self.status_var.set("Form temizlendi.")

    def _build_sample(self) -> dict[str, object]:
        sample: dict[str, object] = {}
        for field_key, cfg in FIELD_CONFIG.items():
            raw_value = self.value_vars[field_key].get().strip().replace(",", ".")
            if not raw_value:
                raise ValueError(f"{cfg['label']} alanı boş bırakılamaz.")
            try:
                numeric_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{cfg['label']} için geçerli bir sayı gir.") from exc
            if not (cfg["min"] <= numeric_value <= cfg["max"]):
                raise ValueError(f"{cfg['label']} değeri {cfg['min']} ile {cfg['max']} arasında olmalı.")
            sample[field_key] = numeric_value

        complaint = self.complaint_var.get().strip()
        if not complaint:
            raise ValueError("Şikayet seçimi yapman gerekiyor.")
        sample["Sikayet"] = complaint
        return sample

    def predict(self) -> None:
        if self.model is None or self.metadata is None:
            messagebox.showwarning("Uyarı", "Model henüz hazır değil.")
            return

        try:
            sample = self._build_sample()
            sample_x = transform_sample(sample, str(DATA_PATH), self.metadata)
            prediction = int(self.model.predict(sample_x)[0])
            label_text = TRIAGE_LABELS.get(prediction, "Tespit edilen seviye için açıklama bulunamadı.")
            self.result_var.set(f"Seviye {prediction}")
            self.result_detail_var.set(f"{label_text}\n\nŞikayet: {sample['Sikayet']}")
            self.result_badge.config(fg=TRIAGE_COLORS.get(prediction, "#f8fafc"))
            self.status_var.set("Tahmin başarıyla üretildi.")
        except Exception as exc:
            self.status_var.set(f"Tahmin hatası: {exc}")
            messagebox.showerror("Hata", str(exc))


def main() -> None:
    root = tk.Tk()
    app = TriageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
