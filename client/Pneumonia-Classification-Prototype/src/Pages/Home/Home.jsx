import React from "react";
import "./home.css";

const Home = () => {
  return (
    <>
      <header>
        <b>Welcome to Our Website</b>
      </header>
      <div className="welcome">
        Early detection of pneumonia is crucial as it can significantly improve
        treatment outcomes and reduce the risk of severe complications.
        Identifying pneumonia in its initial stages allows healthcare providers
        to administer timely and appropriate interventions, preventing the
        condition from progressing to life-threatening levels. With advanced
        tools and technology, early diagnosis not only saves lives but also
        minimizes hospitalization costs and ensures faster recovery for
        patients.
      </div>
      <div className="checking-card">
        <h2>Check Here</h2>
        <div className="checking-card-content">
          <form className="checking-form" action="">
            <label htmlFor="Name"></label>
            <input type="text" placeholder="Enter your name" />
            <label htmlFor="age"></label>
            <input type="text" placeholder="Enter your age" />
            <label htmlFor="upload"></label>
            <input type="file" placeholder="Enter your age" />
            <label htmlFor="submit"></label>
            <input type="submit" placeholder="Enter your age" />
          </form>
          <div className="image-reference">
            <img src="./reference-image.webp" alt="Lung-image" />
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;
