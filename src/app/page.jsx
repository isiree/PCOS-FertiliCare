import styles from './home.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <div className={styles.leftTopContainer} >
        <img src="/pcos2.jpg" alt="Image 1"  />
        <div>
          {/* Content for the left container */}
          <h2>What is PCOS Fertilicare</h2>
          <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          </p>
        </div>
      </div>

      <div className={styles.rightTopContainer}>
        <div>
          {/* Content for the right container */}
          <h2>About PCOS</h2>
          <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit

          </p>
        </div>
        <img src="/pcos2.jpg" alt="Image 2" />
      </div>
    
      <div className={styles.leftBottomContainer}>
        <img src="/pcos2.jpg" alt="Image 1" />
        <div>
          {/* Content for the left container */}
          <h2>Predict Infertility Risk Level</h2>
          <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          </p>
        </div>
      </div>

      <div className={styles.rightBottomContainer}>
        <div>
          {/* Content for the right container */}
          <h2>View Past Results</h2>
          <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          Lorem ipsum dolor sit amet, consectetur adipiscing elit
          </p>
        </div>
        <img src="/pcos2.jpg" alt="Image 2" />
      </div>
      </div>
  );
}