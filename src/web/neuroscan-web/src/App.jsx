import { useState, useCallback } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import UploadZone from './components/UploadZone';
import ApiSettings from './components/ApiSettings';
import ModelSelector from './components/ModelSelector';
import ConfidenceRing from './components/ConfidenceRing';
import ProbabilityBars from './components/ProbabilityBars';
import ImageComparison from './components/ImageComparison';
import { predict } from './api/neuroscan';
import styles from './App.module.css';

const DEFAULT_ENDPOINT = import.meta.env.VITE_API_URL || 'https://the-ace-000-neuroscan-api.hf.space';

const CLASS_LABELS = {
  glioma:     'Glioma',
  meningioma: 'Meningioma',
  pituitary:  'Pituitary',
  notumor:    'No Tumor',
};

export default function App() {
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [model, setModel] = useState('ensemble');
  const [loadedModels, setLoadedModels] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleFile = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError('');
    if (f) {
      setPreviewUrl(URL.createObjectURL(f));
    } else {
      setPreviewUrl(null);
    }
  }, []);

  async function handleSubmit() {
    if (!file) return;
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await predict(endpoint, file, model);
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ??
        err.message ??
        'An error occurred. Check the API endpoint and try again.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }

  const hasResult = result !== null;

  return (
    <>
      <Header />

      <main className={styles.main}>
        <div className={styles.container}>

          {/* Left column: upload + settings */}
          <section className={styles.leftCol}>
            <div className={styles.sectionTitle}>
              <span className={styles.step}>1</span>
              Upload MRI Scan
            </div>
            <UploadZone onFile={handleFile} isLoading={isLoading} />

            <div className={styles.sectionTitle} style={{ marginTop: 24 }}>
              <span className={styles.step}>2</span>
              API Configuration
            </div>
            <ApiSettings endpoint={endpoint} onChange={setEndpoint} onModels={setLoadedModels} />

            <div className={styles.sectionTitle} style={{ marginTop: 24 }}>
              <span className={styles.step}>3</span>
              Select Model
            </div>
            <ModelSelector
              selected={model}
              onChange={setModel}
              loadedModels={loadedModels}
            />

            <button
              className={styles.analyseBtn}
              onClick={handleSubmit}
              disabled={!file || isLoading}
            >
              {isLoading ? (
                <>
                  <span className={styles.btnSpinner} />
                  Analysing&hellip;
                </>
              ) : (
                <>
                  <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                  </svg>
                  Analyse Scan
                </>
              )}
            </button>

            {error && (
              <div className={styles.errorBox}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" style={{ flexShrink: 0 }}>
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
                <span>{error}</span>
              </div>
            )}
          </section>

          {/* Right column: results */}
          <section className={styles.rightCol}>
            {!hasResult && !isLoading && (
              <div className={styles.emptyState}>
                <svg className={styles.emptyIcon} viewBox="0 0 80 80" fill="none">
                  <circle cx="40" cy="40" r="38" stroke="currentColor" strokeWidth="1.5" strokeDasharray="6 4" />
                  <circle cx="40" cy="40" r="18" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="40" cy="40" r="5" fill="currentColor" opacity=".4" />
                  <path d="M40 22v-8M40 66v-8M22 40h-8M66 40h-8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                <p className={styles.emptyTitle}>No results yet</p>
                <p className={styles.emptyHint}>
                  Upload an MRI scan and click <strong>Analyse Scan</strong>
                </p>
              </div>
            )}

            {isLoading && (
              <div className={styles.loadingState}>
                <div className={styles.loadingRing} />
                <p>Running inference&hellip;</p>
                <p className={styles.loadingHint}>EfficientNetB3 &middot; Grad-CAM</p>
              </div>
            )}

            {hasResult && (
              <div className={styles.results}>
                <div className={styles.sectionTitle}>
                  <span className={styles.step} style={{ background: 'var(--blue-600)' }}>4</span>
                  Classification Result
                </div>

                <div className={styles.resultCard}>
                  <div className={styles.resultTop}>
                    <ConfidenceRing
                      prediction={result.class}
                      confidence={result.confidence}
                    />
                    <div className={styles.resultMeta}>
                      <div className={styles.metaRow}>
                        <span className={styles.metaKey}>Prediction</span>
                        <span className={styles.metaVal}>{CLASS_LABELS[result.class] ?? result.class}</span>
                      </div>
                      <div className={styles.metaRow}>
                        <span className={styles.metaKey}>Confidence</span>
                        <span className={styles.metaVal}>
                          {Math.round((result.confidence ?? 0) * 100)}%
                        </span>
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

                <div className={styles.sectionTitle} style={{ marginTop: 24 }}>
                  <span className={styles.step} style={{ background: 'var(--blue-600)' }}>5</span>
                  Scan Visualisation
                </div>
                <ImageComparison original={previewUrl} heatmap={result.heatmap} />
              </div>
            )}
          </section>

        </div>
      </main>

      <Footer />
    </>
  );
}
