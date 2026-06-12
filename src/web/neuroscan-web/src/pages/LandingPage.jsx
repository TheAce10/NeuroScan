import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styles from './LandingPage.module.css';

const SLIDES = [
  {
    label: 'Glioma',
    tag: 'HIGH RISK',
    tagClass: 'tagRed',
    img: '/NeuroScan/samples/glioma.jpg',
    desc: 'Originates from glial cells. Can be high or low grade. Accounts for ~33% of all brain tumours.',
  },
  {
    label: 'Meningioma',
    tag: 'MODERATE RISK',
    tagClass: 'tagOrange',
    img: '/NeuroScan/samples/meningioma.jpg',
    desc: 'Grows from the meninges surrounding the brain. Usually benign but can cause pressure symptoms.',
  },
  {
    label: 'Pituitary',
    tag: 'LOW RISK',
    tagClass: 'tagBlue',
    img: '/NeuroScan/samples/pituitary.jpg',
    desc: 'Tumour of the pituitary gland. Often non-cancerous and manageable with surgery or medication.',
  },
  {
    label: 'No Tumour',
    tag: 'HEALTHY',
    tagClass: 'tagGreen',
    img: '/NeuroScan/samples/notumor.jpg',
    desc: 'No abnormality detected. Normal brain MRI structure with no mass or lesion present.',
  },
];

const MODELS = [
  { name: 'EfficientNetB3', acc: 88.4, glioma: 71, menin: 87 },
  { name: 'VGG16',          acc: 88.9, glioma: 76, menin: 82 },
  { name: 'DenseNet121',    acc: 91.8, glioma: 79, menin: 94 },
];

export default function LandingPage() {
  const [slide, setSlide] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setSlide(s => (s + 1) % SLIDES.length), 3500);
    return () => clearInterval(t);
  }, []);

  const current = SLIDES[slide];

  return (
    <div className={styles.page}>

      {/* ── Nav ── */}
      <nav className={styles.nav}>
        <div className={styles.navInner}>
          <div className={styles.navBrand}>
            <img src="/NeuroScan/ccre-logo.png" alt="CCRE" className={styles.navLogo} />
            <span className={styles.navTitle}>CCRE <span style={{ fontWeight: 400, opacity: 0.45, margin: '0 4px' }}>|</span> NeuroScan</span>
          </div>
          <Link to="/scan" className={styles.navCta}>Launch Scanner</Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <header className={styles.hero}>
        <div className={styles.heroInner}>
          <span className={styles.heroBadge}>AI · Medical Imaging · Research</span>
          <h1 className={styles.heroTitle}>
            Brain Tumour<br />Classification with AI
          </h1>
          <p className={styles.heroSub}>
            An ensemble of three deep learning models classifies MRI scans into four
            categories — with Grad-CAM heatmaps showing exactly where the model looked.
            Built to assess how ready AI is to contribute to medical expertise.
          </p>
          <div className={styles.heroBtns}>
            <Link to="/scan" className={styles.btnPrimary}>Try the Scanner</Link>
            <a href="https://github.com/TheAce10/NeuroScan" target="_blank" rel="noreferrer" className={styles.btnGhost}>
              View on GitHub
            </a>
          </div>
        </div>
      </header>

      {/* ── Demo Slideshow ── */}
      <section className={styles.section}>
        <div className={styles.sectionInner}>
          <h2 className={styles.sectionTitle}>Tumour Classification Demo</h2>
          <p className={styles.sectionSub}>
            NeuroScan identifies four categories from T1-weighted brain MRI scans.
          </p>

          <div className={styles.slideshow}>
            {/* Scan panel */}
            <div className={styles.slideImg}>
              <img
                key={slide}
                src={current.img}
                alt={current.label}
                className={styles.scanImg}
                onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
              />
              <div className={styles.scanPlaceholder} style={{ display: 'none' }}>
                <svg viewBox="0 0 80 80" fill="none" width="60" height="60">
                  <circle cx="40" cy="40" r="38" stroke="currentColor" strokeWidth="1.5" strokeDasharray="6 4" />
                  <circle cx="40" cy="40" r="18" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="40" cy="40" r="5" fill="currentColor" opacity=".4" />
                </svg>
                <span>Sample image</span>
              </div>
              <div className={`${styles.slideTag} ${styles[current.tagClass]}`}>{current.tag}</div>
            </div>

            {/* Info panel */}
            <div className={styles.slideInfo}>
              <h3 className={styles.slideLabel}>{current.label}</h3>
              <p className={styles.slideDesc}>{current.desc}</p>
              <div className={styles.slideDots}>
                {SLIDES.map((_, i) => (
                  <button
                    key={i}
                    className={`${styles.dot} ${i === slide ? styles.dotActive : ''}`}
                    onClick={() => setSlide(i)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Model Performance ── */}
      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className={styles.sectionInner}>
          <h2 className={styles.sectionTitle}>Model Performance</h2>
          <p className={styles.sectionSub}>
            All models trained on the Brain Tumor MRI Dataset via Kaggle T4 GPU.
            DenseNet121 uses focal loss to address the glioma/meningioma boundary problem.
          </p>
          <div className={styles.modelCards}>
            {MODELS.map(m => (
              <div key={m.name} className={`${styles.modelCard} ${m.name === 'DenseNet121' ? styles.modelCardBest : ''}`}>
                {m.name === 'DenseNet121' && <span className={styles.bestBadge}>Best Model</span>}
                <h3 className={styles.modelName}>{m.name}</h3>
                <div className={styles.modelAcc}>{m.acc}%</div>
                <div className={styles.modelAccLabel}>Overall Accuracy</div>
                <div className={styles.modelStats}>
                  <div className={styles.modelStat}>
                    <span className={styles.statVal}>{m.glioma}%</span>
                    <span className={styles.statLabel}>Glioma Recall</span>
                  </div>
                  <div className={styles.modelStat}>
                    <span className={styles.statVal}>{m.menin}%</span>
                    <span className={styles.statLabel}>Meningioma Recall</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className={styles.ctaSection}>
        <div className={styles.sectionInner} style={{ textAlign: 'center' }}>
          <h2 className={styles.ctaTitle}>Ready to classify a scan?</h2>
          <p className={styles.ctaSub}>Upload an MRI image and get a prediction with visualisation in seconds.</p>
          <Link to="/scan" className={styles.btnPrimary} style={{ display: 'inline-flex' }}>
            Launch Scanner
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <div className={styles.sectionInner}>
          <span>NeuroScan v2.0 · Centre for Computational Research and Education</span>
          <a href="https://github.com/TheAce10/NeuroScan" target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </footer>

    </div>
  );
}
