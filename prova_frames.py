import dv_processing as dv
import cv2

# Sostituisci con l'IP e la porta del tuo tcp_server in DV
indirizzo_ip = "127.0.0.1"
porta_tcp = 7778

print(f"Connessione a {indirizzo_ip}:{porta_tcp}...")
reader = dv.io.NetworkReader(indirizzo_ip, porta_tcp)

# Controlliamo cosa sta effettivamente inviando il server
has_events = reader.isEventStreamAvailable()
has_frames = reader.isFrameStreamAvailable()

print(f"Stream eventi trovato: {has_events}")
print(f"Stream frame trovato: {has_frames}")

if not (has_events or has_frames):
    raise RuntimeError("Il server non sta inviando né eventi né frame. Controlla il DV!")

cv2.namedWindow("Frame Stream", cv2.WINDOW_NORMAL)

# Ciclo principale
while reader.isRunning():
    
    # 1. Leggi i Frame se disponibili
    if has_frames:
        frame = reader.getNextFrame()
        if frame is not None:
            # frame.image è l'array NumPy contenente i pixel
            cv2.imshow("Frame Stream", frame.image)
            
    # 2. Leggi gli Eventi se disponibili
    if has_events:
        events = reader.getNextEventBatch()
        if events is not None and events.size() > 0:
            # Fai ciò che vuoi con gli eventi (es. filtraggio, accumulo, ecc.)
            # events è un oggetto dv.EventStore
            pass 
            # print(f"Ricevuto batch di {events.size()} eventi")

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()