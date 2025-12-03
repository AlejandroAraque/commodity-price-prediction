import torch
import os

# --- PON AQUÍ LOS NOMBRES DE TUS ARCHIVOS ---
FILE_A = "checkpoints/V10_Clasificacion-epoch=38-val_acc=0.5502.ckpt"
FILE_B = "checkpoints/V11_Gold-epoch=38-val_acc=0.5502.ckpt"

def compare_models(path_a, path_b):
    print(f"🕵️‍♂️ COMPARANDO MODELOS:\n A: {path_a}\n B: {path_b}\n")
    
    # 1. Comprobación de Archivo Físico
    size_a = os.path.getsize(path_a)
    size_b = os.path.getsize(path_b)
    print(f"💾 TAMAÑO EN DISCO:")
    print(f"   A: {size_a} bytes")
    print(f"   B: {size_b} bytes")
    if size_a == size_b:
        print("   ⚠️ ¡ALERTA! Tienen el mismo tamaño exacto. Podrían ser el mismo archivo copiado.")
    else:
        print("   ✅ Tamaños diferentes. Son entrenamientos distintos.")

    # 2. Cargar Checkpoints
    # map_location='cpu' es vital para no necesitar GPU
    ckpt_a = torch.load(path_a, map_location='cpu')
    ckpt_b = torch.load(path_b, map_location='cpu')

    # 3. Comparar Hiperparámetros
    hparams_a = ckpt_a.get('hyper_parameters', {})
    hparams_b = ckpt_b.get('hyper_parameters', {})
    
    print(f"\n⚙️ HIPERPARÁMETROS (Diferencias):")
    all_keys = set(hparams_a.keys()) | set(hparams_b.keys())
    diff_found = False
    for key in all_keys:
        val_a = hparams_a.get(key, "N/A")
        val_b = hparams_b.get(key, "N/A")
        if val_a != val_b:
            print(f"   🔴 {key}: A={val_a} vs B={val_b}")
            diff_found = True
            
    if not diff_found:
        print("   ✅ Hiperparámetros idénticos.")

    # 4. Comparar Pesos (La Inteligencia Real)
    state_a = ckpt_a['state_dict']
    state_b = ckpt_b['state_dict']
    
    # Comparamos la primera capa de pesos para ver si son matemáticamente iguales
    first_layer_key = list(state_a.keys())[0]
    weights_a = state_a[first_layer_key]
    weights_b = state_b[first_layer_key]
    
    print(f"\n🧠 CEREBRO (Pesos):")
    if torch.equal(weights_a, weights_b):
        print("   ⚠️ ¡ALERTA ROJA! Los pesos son IDÉNTICOS. Es el mismo modelo renombrado.")
    else:
        print("   ✅ Los pesos son DIFERENTES. Aprendieron cosas distintas.")
        # Ver cuánto difieren
        diff = (weights_a - weights_b).abs().mean()
        print(f"   Diferencia media en primera capa: {diff:.6f}")

if __name__ == "__main__":
    compare_models(FILE_A, FILE_B)