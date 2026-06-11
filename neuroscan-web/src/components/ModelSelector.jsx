import styles from './ModelSelector.module.css';

const MODELS = [
  { id: 'ensemble',        label: 'Ensemble',        note: 'All loaded models (recommended)' },
  { id: 'efficientnet_b3', label: 'EfficientNetB3',  note: '88.4% accuracy' },
  { id: 'vgg16',           label: 'VGG16',            note: '88.9% accuracy' },
  { id: 'densenet121',     label: 'DenseNet121',      note: 'Focal loss — best glioma/meningioma' },
];

export default function ModelSelector({ selected, onChange, loadedModels }) {
  return (
    <div className={styles.wrap}>
      {MODELS.map(m => {
        const available = m.id === 'ensemble'
          ? (!loadedModels || loadedModels.length >= 2)
          : (!loadedModels || loadedModels.includes(m.id));
        const active    = selected === m.id;
        return (
          <button
            key={m.id}
            className={[
              styles.btn,
              active     ? styles.active     : '',
              !available ? styles.unavailable : '',
            ].join(' ')}
            onClick={() => available && onChange(m.id)}
            disabled={!available}
            title={available ? m.label : `${m.label} — not loaded`}
          >
            <span className={styles.name}>{m.label}</span>
            <span className={styles.note}>{available ? m.note : 'not loaded'}</span>
          </button>
        );
      })}
    </div>
  );
}
