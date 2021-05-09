// grafika-toolkit.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cstdlib>
#include "framework.h"

int main() {


    /*
        Ha valami nincs benne a kódban, akkor nagyon nagy segítség a feladatokhoz ez a doksi: 
        https://docs.google.com/document/d/1k7gOzvxJvXdFESTenn-g-9HVq_B3M62rUi269raxuz0/edit?fbclid=IwAR1_9V4eIfC9QuKQyNLcQFX77b_dx6OOn26-paU-1kuRI2-781XHOdugaf0#heading=h.54ubfmubwg0u
    
    */
    /*
    Peldakodok:
    //1.es kviz
    //
    pontEgyenes(vec2(-5, 4), vec3(3, 4, 5));
    //
    varos(0, 41, 0, 12,6000);
    //
    hyperbolicD(vec3(0,0,0), vec3(0,0,0));

    //2. es kviz
    //
    BezierCurve bc;
    bc.AddControlPoint(vec3(3,3));
    bc.AddControlPoint(vec3(5,1));
    bc.AddControlPoint(vec3(6,9));
    tmp3 = bc.r(1.0);
    printf("BC: X: %3.2f Y: %3.2f Z: %3.2f\n", tmp3.x, tmp3.y, tmp3.z);

    //Nem mindig jó :(
    //
    CatmullRom cr;
    cr.AddControlPoint(vec2(5, 5), 0);
    cr.AddControlPoint(vec2(2, 4), 1);
    cr.AddControlPoint(vec2(5, 3), 2);
    cr.AddControlPoint(vec2(2, 4), 3);
    //Ha X: 3.5 akkor 3.6875 xd
    tmp3 = cr.r(1.5);
    printf("CR: X: %3.2f Y: %3.2f Z: %3.2f\n", tmp3.x, tmp3.y, tmp3.z);

    //
    LagrangeCurve lc;
    lc.AddControlPoint(vec3(5,1));
    lc.AddControlPoint(vec3(2,10));
    lc.AddControlPoint(vec3(7,7));
    tmp3 = lc.r(1.0);
    printf("LC: X: %3.2f Y: %3.2f Z: %3.2f\n", tmp3.x, tmp3.y, tmp3.z);

    //3. kviz
    //
    affin(vec2(3, 3), 0, 0, 0, 0, 0, 0);

    //
    //A FELADATBAN MEGADOTT KOORD AZ UTOLSO (4. koord) LEGYEN S = q.w 
    vec4 q = vec4(0, 0, sqrt(2) / 2, sqrt(2) / 2);
    vec4 u = vec4(9, 0, 0, 0);
    vec4 res = qmul(q, u);
    vec4 qinv = vec4(0, 0, -0.707106781186548, -0.707106781186548);
    tmp4 = qmul(res,qinv);
    //Ha inverz van akkor matlabbal ki lehet szamolni: qinv = quatinv([1 0 1 0])
    printf("S: %3.2f X: %3.2f Y: %3.2f Z: %3.2f\n", tmp4.w, tmp4.x, tmp4.y, tmp4.z);

    //
    //vagy kvat szorzas: https://www.vcalc.com/wiki/vCalc/Quaternion+Multiplication

    //
    ketEgyenesImplicit(vec3(4, 5, 2.5), vec3(12, 15, 14));

    //
    szakaszketvegpont(vec4(-1,5,0,1), vec4(7,6,0,-1));


    //4.kviz
    //
    camera2d(vec2(168, 968), 14, 7, vec2(221, 16));

    //
    DDA(29, 21, 86, 39, 16);

    //6. kviz
    //
    haromszogMVP(vec3(0,0,-1), vec2(0,0), vec3(0,1,-0.5), vec2(0,1), vec3(1,0,-0.5), vec2(1,0));


    //7. kviz
    //
    sugarsurusegDiffuz(vec3(0, 0, 1), vec3(0, 3, 4), 9,0);

    //
    sugarsurusegSpekularis(vec3(0,0,1),vec3(0,3,4),4,sqrt(2),vec3(0,4,3),3);
 
    //
    fenysugarKozegbol(0.7);

    //
    F(1.2, 0);

    //9. kviz

    //
    gltriangles(8);

    //
    gltriangleStrip(7);

    //
    parameteresFelulet(9, 7.6, 5,
                       9.6, 5, 10,
                       5.1, 5.6, 2.9,
                       0.4, 0.6);

    //
    haromszogCsucs(vec3(68,34,0.7),vec3(19,89,0.1),vec3(86,61,0.7));

    //10. kviz

    //
    testElek(3, 10, 18);
 
    //
    parsecPersec(vec3(5,2,4), vec3(3,2,1), vec3(3,2,3), vec3(3,4,2), 100);

    //
    */

    vec3 tmp3;
    vec4 tmp4;

    //FPS(vec3(0, 0, 0), vec3(6, 8, 0), vec3(12, -9, 0));
    billBoard(vec3(4,2,7),vec3(1,2,3),vec3(0,1,0));

}
