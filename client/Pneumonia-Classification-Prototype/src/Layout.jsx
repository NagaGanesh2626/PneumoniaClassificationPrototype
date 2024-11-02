import Navbar from "./Components/Navbar/Navbar";
import { Outlet } from "react-router-dom";
import Background from "./Components/Background/Background";

const Layout = () => {
  return (
    <>
      <Navbar />
      <Background>
        <Outlet />
      </Background>
    </>
  );
};

export default Layout;
