import { useState, useCallback } from 'react';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';
import UploadZone from '../components/UploadZone';
import ModelSelector from '../components/ModelSelector';
import ConfidenceRing from '../components/ConfidenceRing';
import ProbabilityBars from '../components/ProbabilityBars';
import ImageComparison from '../components/ImageComparison';
import { predict } from '../api/neuroscan';
import styles from './ScannerPage.module.css';

const API_URL = import.meta.env.VITE_API_URL || 'https://the-ace-000-neuroscan-api.hf.space';

const CLASS_LABELS = {
  glioma:     'Glioma',
  meningioma: 'Meningioma',
  pituitary:  'Pituitary',
  notumor:    'No Tumor',
};

export default function ScannerPage() {
  const [model, setModel] = useState('ensemble');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [warn, setWarn] = useState({ text: '', strong: false });

  const handleFile = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError('');
    setWarn({ text: '', strong: false });
    if (!f) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);

    const img = new window.Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 100; canvas.height = 100;
      canvas.getContext('2d').drawImage(img, 0, 0, 100, 100);
      const data = canvas.getContext('2d').getImageData(0, 0, 100, 100).data;
      let totalSat = 0, count = 0;
      for (let i = 0; i < data.length; i += 16) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        totalSat += max === 0 ? 0 : (max - min) / max;
        count++;
      }
      const avgSat = totalSat / count;

      if (avgSat > 0.15) {
        setWarn({
          text: 'This image appears to be in colour. Brain MRI scans are grayscale — any result produced will be meaningless. Please upload a T1 or T2-weighted MRI scan.',
          strong: true,
        });
      } else {
        const ratio = Math.max(img.width, img.height) / Math.min(img.width, img.height);
        if (ratio > 1.4) {
          setWarn({
            text: 'Portrait-format image detected. It will be center-cropped to square before analysis.',
            strong: false,
          });
        }
      }
    };
    img.src = url;
  }, []);

  async function preprocessToSquare(f) {
    return new Promise((resolve) => {
      const img = new window.Image();
      const url = URL.createObjectURL(f);
      img.onload = () => {
        const size = Math.min(img.width, img.height);
        const sx = (img.width - size) / 2;
        const sy = (img.height - size) / 2;
        const canvas = document.createElement('canvas');
        canvas.width = 300; canvas.height = 300;
        canvas.getContext('2d').drawImage(img, sx, sy, size, size, 0, 0, 300, 300);
        URL.revokeObjectURL(url);
        canvas.toBlob(resolve, 'image/jpeg', 0.92);
      };
      img.src = url;
    });
  }

  async function handleSubmit() {
    if (!file) return;
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const processed = await preprocessToSquare(file);
      const data = await predict(API_URL, processed, model);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ??
        err.message ??
        'An error occurred. The API may be starting up — please try again in a moment.'
      );
    } finally {
      setIsLoading(false);
    }
  }

  const lowConfidence = result && (result.confidence ?? 0) < 0.65;

  return (
    <>
      <NavBar />

      <main className={styles.main}>

        {/* ── Inputs row ── */}
        <div className={styles.inputsGrid}>
          <div className={styles.inputLeft}>
            <div className={styles.sectionTitle}>
              <span className={styles.step}>1</span>
              Upload MRI Scan
            </div>
            <UploadZone onFile={handleFile} isLoading={isLoading} />
          </div>

          <div className={styles.inputRight}>
            <div className={styles.sectionTitle}>
              <span className={styles.step}>2</span>
              Select Model
            </div>
            <ModelSelector selected={model} onChange={setModel} disabled={isLoading} />

            <button
              className={styles.analyseBtn}
              onClick={handleSubmit}
              disabled={!file || isLoading}
            >
              {isLoading ? (
                <><span className={styles.btnSpinner} /> Analysing&hellip;</>
              ) : (
                <>
                  <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                  </svg>
                  Analyse Scan
                </>
              )}
            </button>

            {warn.text && (
              <div className={warn.strong ? styles.warnBoxStrong : styles.warnBox}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" style={{ flexShrink: 0, marginTop: 1 }}>
                  <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
                <span>{warn.text}</span>
              </div>
            )}

            {error && (
              <div className={styles.errorBox}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" style={{ flexShrink: 0, marginTop: 1 }}>
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* ── Loading ── */}
        {isLoading && (
          <div className={styles.loadingState}>
            <div className={styles.loadingRing} />
            <p>Running inference&hellip;</p>
            <p className={styles.loadingHint}>This may take 15–60 seconds on first run</p>
          </div>
        )}

        {/* ── Results row ── */}
        {result && !isLoading && (
          <>
            <hr className={styles.divider} />
            <div className={styles.resultsGrid}>

              <div>
                <div className={styles.sectionTitle}>
                  <span className={styles.step} style={{ background: 'var(--blue-600)' }}>3</span>
                  Classification Result
                </div>
                <div className={styles.resultCard}>
                  <div className={styles.resultTop}>
                    <ConfidenceRing prediction={result.class} confidence={result.confidence} />
                    <div className={styles.resultMeta}>
                      <div className={styles.metaRow}>
                        <span className={styles.metaKey}>Prediction</span>
                        <span className={styles.metaVal}>{CLASS_LABELS[result.class] ?? result.class}</span>
                      </div>
                      <div className={styles.metaRow}>
                        <span className={styles.metaKey}>Confidence</span>
                        <span className={styles.metaVal}>{Math.round((result.confidence ?? 0) * 100)}%</span>
                      </div>
                      {result.model && (
                        <div className={styles.metaRow}>
                          <span className={styles.metaKey}>Model</span>
                          <span className={styles.metaVal}>{result.model}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <ProbabilityBars scores={result.scores} prediction={result.class} />
                </div>

                {lowConfidence && (
                  <div className={styles.lowConfidenceBanner}>
                    <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" style={{ flexShrink: 0, marginTop: 1 }}>
                      <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <strong>Low confidence ({Math.round((result.confidence ?? 0) * 100)}%)</strong>
                      <span> — the model is uncertain. This may indicate an ambiguous scan, unusual presentation, or a non-MRI image. Do not rely on this result.</span>
                    </div>
                  </div>
                )}
              </div>

              <div>
                <div className={styles.sectionTitle}>
                  <span className={styles.step} style={{ background: 'var(--blue-600)' }}>4</span>
                  Scan Visualisation
                </div>
                <ImageComparison original={previewUrl} heatmap={result.heatmap} />
              </div>

            </div>
          </>
        )}

      </main>

      <Footer />
    </>
  );
}
