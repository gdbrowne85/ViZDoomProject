#include "ViziaDoomGamePython.h"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
namespace Vizia {

    using boost::python::tuple;
    using boost::python::api::object;
    using boost::python::numeric::array;
#define PY_NONE object()

    DoomGamePython::DoomGamePython() {
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
        import_array();
        
        //boost::numpy::initialize();
        this->numpyImage = NULL;
        this->numpyVars = NULL;
    }
    DoomGamePython::~DoomGamePython() {
        delete this->numpyImage;
        delete this->numpyVars;
        this->numpyImage = NULL;
        this->numpyVars = NULL;
    }

    bool DoomGamePython::init() {
        bool initSuccess = DoomGame::init();

        if (initSuccess) {

            int channels = this->getScreenChannels();
            int x = this->getScreenWidth();
            int y = this->getScreenHeight();
            npy_intp imageShape[3];
            switch(this->getScreenFormat())
            {
                case CRCGCB:
                case CRCGCBCA:
                case CBCGCR:
                case CBCGCRCA:
                    imageShape[0] = channels;
                    imageShape[1] = x;
                    imageShape[2] = y;
                    break;
                default:
                    imageShape[0] = x;
                    imageShape[1] = y;
                    imageShape[2] = channels;
            }
            PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, this->doomController->getScreen());
            this->numpyImageHandle = boost::python::handle<>(img);
            this->numpyImage = new array(this->numpyImageHandle);
            if (this->state.vars.size() > 0) {
                npy_intp varLen = this->state.vars.size();
                PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, this->state.vars.data());
                this->numpyVarsHandle = boost::python::handle<>(vars);
                this->numpyVars = new array(this->numpyVarsHandle);
            }

        }
        return initSuccess;
    }

    float DoomGamePython::makeAction(boost::python::list actionList) {
        // TODO what if isFinished()?
        int listLength = boost::python::len(actionList);
        if( listLength != this->getActionFormat())
        {
            cerr<<"Incorrect action length: "<<listLength<<" Should be: "<<this->getActionFormat()<<endl;
            //maybe throw something?
            return 0;
        }
        std::vector<bool> action = std::vector<bool>(listLength);
        for (int i = 0; i < listLength; i++) {
            action[i] = boost::python::extract<bool>(actionList[i]);
        }
        return DoomGame::makeAction(action);
        
    }

    DoomGamePython::PythonState DoomGamePython::getState() {
        if (isEpisodeFinished()) {
            return DoomGamePython::PythonState(this->state.number);
        }
        if (this->state.vars.size() > 0) {
            return DoomGamePython::PythonState(this->state.number, this->numpyImage->copy(), this->numpyVars->copy());
        }
        else {
            return DoomGamePython::PythonState(this->state.number, this->numpyImage->copy());
        }


    }

/* not sure if we need this */
    object DoomGamePython::getLastAction() {
        //TODO
        return PY_NONE;
    }
}