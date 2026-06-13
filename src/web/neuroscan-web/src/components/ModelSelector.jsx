import styles from './ModelSelector.module.css';

const MODELS = [
  { id: 'ensemble',        label: 'Ensemble',       note: 'Combined models (recommended)' },
  { id: 'efficientnet_b3', label: 'EfficientNetB3', note: '92.0% accuracy' },
  { id: 'vgg16',           label: 'VGG16',          note: '93.0% accuracy · Best' },
  { id: 'densenet121',     label: 'DenseNet121',    note: '91.0% accuracy' },
];

export default function ModelSelector({ selected, onChange, disabled }) {
  return (
    <div className={styles.wrap}>
      {MODELS.map(m => {
        const isSelected = selected === m.id;
        const isInactive = disabled && !isSelected;
        return (
          <button
            key={m.id}
            className={[
              styles.btn,
              isSelected ? styles.active : '',
              isInactive ? styles.inactive : '',
            ].join(' ')}
            onClick={() => !disabled && onChange(m.id)}
            disabled={isInactive}
          >
            <span className={styles.name}>{m.label}</span>
            <span className={styles.note}>
              {isSelected && disabled ? 'Running…' : m.note}
            </span>
          </button>
        );
      })}
    </div>
  );
}
