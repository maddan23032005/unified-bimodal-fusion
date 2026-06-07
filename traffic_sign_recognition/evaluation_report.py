import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from ultralytics import YOLO

def main():
    # ---------- Load metrics from results.csv ----------
    df = pd.read_csv("runs/detect/traffic_sign/results.csv")
    final = df.iloc[-1]

    precision = final['metrics/precision(B)']
    recall = final['metrics/recall(B)']
    map50 = final['metrics/mAP50(B)']
    map5095 = final['metrics/mAP50-95(B)']
    box_loss = final['val/box_loss']
    cls_loss = final['val/cls_loss']

    # ---------- Print key metrics ----------
    print("\n--- YOLOv8n Final Evaluation Report ---")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"mAP@0.5              : {map50:.4f}")
    print(f"mAP@0.5:0.95         : {map5095:.4f}")
    print(f"Validation Box Loss  : {box_loss:.4f}")
    print(f"Validation Class Loss: {cls_loss:.4f}")

    # ---------- Plot Training Curves ----------
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("YOLOv8n Evaluation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    metrics_plot_path = "metrics_plot.png"
    plt.savefig(metrics_plot_path)
    plt.close()

    # ---------- Load model and run validation ----------
    model = YOLO("runs/detect/traffic_sign/weights/best.pt")
    results = model.val()

    # ---------- Get and Plot Confusion Matrix ----------
    cm = results.confusion_matrix.matrix
    class_names = list(results.names.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    conf_matrix_path = "confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()

    # ---------- Generate PDF Report ----------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "YOLOv8n Model Evaluation Report", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Precision            : {precision:.4f}", ln=True)
    pdf.cell(0, 10, f"Recall               : {recall:.4f}", ln=True)
    pdf.cell(0, 10, f"mAP@0.5              : {map50:.4f}", ln=True)
    pdf.cell(0, 10, f"mAP@0.5:0.95         : {map5095:.4f}", ln=True)
    pdf.cell(0, 10, f"Validation Box Loss  : {box_loss:.4f}", ln=True)
    pdf.cell(0, 10, f"Validation Class Loss: {cls_loss:.4f}", ln=True)

    # ---------- Add Metrics Plot ----------
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Training Metric Curves", ln=True)
    if os.path.exists(metrics_plot_path):
        pdf.image(metrics_plot_path, x=10, y=None, w=180)

    # ---------- Add Confusion Matrix ----------
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Confusion Matrix", ln=True)
    if os.path.exists(conf_matrix_path):
        pdf.image(conf_matrix_path, x=10, y=None, w=180)

    pdf_output = "YOLOv8n_Complete_Report.pdf"
    pdf.output(pdf_output)
    print(f"\n📄 Full report saved as: {pdf_output}")

# ---------- Required for Windows ----------
if __name__ == "__main__":
    main()
