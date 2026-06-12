import styles from './ModelSelector.module.css';

const MODELS = [
  { id: 'ensemble',        label: 'Ensemble',       note: 'Combined models (recommended)' },
  { id: 'efficientnet_b3', label: 'EfficientNetB3', note: '88.4% accuracy' },
  { id: 'vgg16',           label: 'VGG16',           note: '88.9% accuracy' },
  { id: 'densenet121',     label: 'DenseNet121',     note: '91.8% accuracy' },
];

export default function ModelSelector({ selected, onChange }) {
  return (
    <div className={styles.wrap}>
      {MODELS.map(m => (
        <button
          key={m.id}
          className={[styles.btn, selected === m.id ? styles.active : ''].join(' ')}
          onClick={() => onChange(m.id)}
        >
          <span className={styles.name}>{m.label}</span>
          <span className={styles.note}>{m.note}</span>
        </button>
      ))}
    </div>
  );
}
