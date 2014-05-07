
#include "G4Polyhedron.hh"
#include "G4VGraphicsScene.hh"
#include "G4VisManager.hh"

#include "GateImageBox.hh"
#include "GateImage.hh"

#include <typeinfo>

GateImageBox::GateImageBox(const GateImage & image, const G4String & name) : G4Box(name, image.GetHalfSize().x(), image.GetHalfSize().y(), image.GetHalfSize().z()) {

#ifdef GATEIMAGEBOX_USE_OPENGL
    G4VisManager * pVisManager = dynamic_cast<G4VisManager *>(G4VVisManager::GetConcreteInstance());
    if(pVisManager) {

        G4OpenGLSceneHandler * opengl = dynamic_cast<G4OpenGLSceneHandler *>( pVisManager->GetCurrentSceneHandler() );
        if(opengl) {
            initOpenGLTextures(image, image.GetResolution().x() * 0.5, image.GetResolution().y() * 0.5, image.GetResolution().z() * 0.5);
        }

    }
#endif

}

GateImageBox::~GateImageBox() {}

void GateImageBox::DescribeYourselfTo(G4VGraphicsScene& scene) const{
#ifdef GATEIMAGEBOX_USE_OPENGL
    try
    {
        G4OpenGLSceneHandler& opengl = dynamic_cast<G4OpenGLSceneHandler&>(scene);

        scene.AddSolid (*this);
        DescribeYourselfTo(opengl);
    }
    catch(std::bad_cast exp)
    {
        scene.AddSolid (*this);
    }
#else
    scene.AddSolid (*this);
#endif

}

#ifdef GATEIMAGEBOX_USE_OPENGL
void GateImageBox::DescribeYourselfTo(G4OpenGLSceneHandler& scene) const{

    scene.BeginPrimitives(scene.GetObjectTransformation());

    GLfloat xHalfLength = GetXHalfLength();
    GLfloat yHalfLength = GetYHalfLength();
    GLfloat zHalfLength = GetZHalfLength();

    GLfloat x = position_x * xHalfLength * 2 / resolution_x - xHalfLength;
    GLfloat y = position_y * yHalfLength * 2 / resolution_y - yHalfLength;
    GLfloat z = position_z * zHalfLength * 2 / resolution_z - zHalfLength;

    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    GLfloat * color = new GLfloat[4];
    glGetFloatv(GL_CURRENT_COLOR, color);
    
    glColor3f(1.f, 1.0f, 1.0f);

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, texture_xy);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0); glVertex3f(-xHalfLength, -yHalfLength, z);
    glTexCoord2d(0, 1); glVertex3f(-xHalfLength,  yHalfLength, z);
    glTexCoord2d(1, 1); glVertex3f( xHalfLength,  yHalfLength, z);
    glTexCoord2d(1, 0); glVertex3f( xHalfLength, -yHalfLength, z);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, texture_yz);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0); glVertex3f(x, -yHalfLength, -zHalfLength);
    glTexCoord2d(1, 0); glVertex3f(x,  yHalfLength, -zHalfLength);
    glTexCoord2d(1, 1); glVertex3f(x,  yHalfLength,  zHalfLength);
    glTexCoord2d(0, 1); glVertex3f(x, -yHalfLength,  zHalfLength);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, texture_xz);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0); glVertex3f(-xHalfLength, y, -zHalfLength);
    glTexCoord2d(0, 1); glVertex3f(-xHalfLength, y,  zHalfLength);
    glTexCoord2d(1, 1); glVertex3f( xHalfLength, y,  zHalfLength);
    glTexCoord2d(1, 0); glVertex3f( xHalfLength, y, -zHalfLength);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-xHalfLength, -yHalfLength, z);
    glVertex3f(-xHalfLength,  yHalfLength, z);
    glVertex3f( xHalfLength,  yHalfLength, z);
    glVertex3f( xHalfLength, -yHalfLength, z);
    glEnd();
    
    glBegin(GL_LINE_LOOP);
    glVertex3f(x, -yHalfLength, -zHalfLength);
    glVertex3f(x,  yHalfLength, -zHalfLength);
    glVertex3f(x,  yHalfLength,  zHalfLength);
    glVertex3f(x, -yHalfLength,  zHalfLength);
    glEnd();
    
    glBegin(GL_LINE_LOOP);
    glVertex3f(-xHalfLength, y, -zHalfLength);
    glVertex3f(-xHalfLength, y,  zHalfLength);
    glVertex3f( xHalfLength, y,  zHalfLength);
    glVertex3f( xHalfLength, y, -zHalfLength);
    glEnd();
    
    glColor3fv(color);
    delete color;
    
    glPopAttrib();

    scene.EndPrimitives();
}

std::vector<GateImage::PixelType> GateImageBox::getXYSlice(const GateImage & image, const size_t z) const{
    std::vector<GateImage::PixelType> slice;
    slice.reserve(image.GetResolution().x() * image.GetResolution().y());
    for(size_t y = 0; y < image.GetResolution().y(); y++) {
        for(size_t x = 0; x < image.GetResolution().x(); x++) {
            GateImage::PixelType value = image.GetValue(x, y, z);
            slice.push_back(value);
        }
    }

    return slice;
}

std::vector<GateImage::PixelType> GateImageBox::getXZSlice(const GateImage & image, const size_t y) const{
    std::vector<GateImage::PixelType> slice;
    slice.reserve(image.GetResolution().x() * image.GetResolution().z());
    for(size_t z = 0; z < image.GetResolution().z(); z++) {
        for(size_t x = 0; x < image.GetResolution().x(); x++) {
            GateImage::PixelType value = image.GetValue(x, y, z);
            slice.push_back(value);
        }
    }

    return slice;
}

std::vector<GateImage::PixelType> GateImageBox::getYZSlice(const GateImage & image, const size_t x) const{
    std::vector<GateImage::PixelType> slice;
    slice.reserve(image.GetResolution().y() * image.GetResolution().z());

    for(size_t z = 0; z < image.GetResolution().z(); z++) {
        for(size_t y = 0; y < image.GetResolution().y(); y++) {
            GateImage::PixelType value = image.GetValue(x, y, z);
            slice.push_back(value);
        }
    }

    return slice;
}

GLubyte * GateImageBox::convertToRGB(std::vector<GateImage::PixelType> slice, GateImage::PixelType min, GateImage::PixelType max) const{
    GLubyte * rgb = new GLubyte[slice.size() * 3];

    GateImage::PixelType interval = max - min;
    int i = 0;
    for(std::vector<GateImage::PixelType>::iterator it = slice.begin(); it != slice.end(); ++it) {
        GateImage::PixelType pixel = *it - min;
        pixel /= interval;
        pixel *= std::numeric_limits<GLubyte>::max();

        GLubyte gray = static_cast<GLubyte>(pixel);
        rgb[i] = gray;
        rgb[i+1] = gray;
        rgb[i+2] = gray;
        i += 3;
    }

    return rgb;
}

GLuint GateImageBox::genOpenGLTexture(const GLubyte * rgb, int width, int height) const{
    GLuint texture;

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*) rgb);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture;
}

void GateImageBox::initOpenGLTextures(const GateImage & image, const size_t x, const size_t y, const size_t z) {
    GateImage::PixelType min = image.GetMinValue();
    GateImage::PixelType max = image.GetMaxValue();

    resolution_x = image.GetResolution().x();
    resolution_y = image.GetResolution().y();
    resolution_z = image.GetResolution().z();

    position_x = (x < resolution_x) ? x : resolution_x * 0.5;
    position_y = (y < resolution_y) ? y : resolution_y * 0.5;
    position_z = (z < resolution_z) ? z : resolution_z * 0.5;

    {
        std::vector<GateImage::PixelType> sliceXY = getXYSlice(image, position_z);
        GLubyte * rgb = convertToRGB(sliceXY, min, max);
        texture_xy = genOpenGLTexture(rgb, image.GetResolution().x(), image.GetResolution().y());
        delete [] rgb;
    }

    {
        std::vector<GateImage::PixelType> sliceXZ = getXZSlice(image, position_y);
        GLubyte * rgb = convertToRGB(sliceXZ, min, max);
        texture_xz = genOpenGLTexture(rgb, image.GetResolution().x(), image.GetResolution().z());
        delete [] rgb;
    }

    {
        std::vector<GateImage::PixelType> sliceYZ = getYZSlice(image, position_x);
        GLubyte * rgb = convertToRGB(sliceYZ, min, max);
        texture_yz = genOpenGLTexture(rgb, image.GetResolution().y(), image.GetResolution().z());
        delete [] rgb;
    }
}

#endif
