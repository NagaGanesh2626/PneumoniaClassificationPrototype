import "./Navbar.css";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <>
      <nav className="nav-bar">
        <div className="logo">
          <Link to="/">LungCheck</Link>
        </div>
        <div className="pages">
          <div className="home">
            <Link to="/">Home</Link>
          </div>
          <div className="about">
            <Link to="/about">About</Link>
          </div>
          <div className="textor">Check Here</div>
          <div className="contact">Contact Us</div>
        </div>
      </nav>
    </>
  );
}

export default Navbar;
