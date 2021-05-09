#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
# define M_PI           3.14159265358979323846 

//--------------------------
struct vec2 {
	//--------------------------
	float x, y;

	vec2(float x0 = 0, float y0 = 0) { x = x0; y = y0; }
	vec2 operator*(float a) const { return vec2(x * a, y * a); }
	vec2 operator/(float a) const { return vec2(x / a, y / a); }
	vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
	vec2 operator-() const { return vec2(-x, -y); }
};

inline float dot(const vec2& v1, const vec2& v2) {
	return (v1.x * v2.x + v1.y * v2.y);
}

inline float length(const vec2& v) { return sqrtf(dot(v, v)); }

inline vec2 normalize(const vec2& v) { return v * (1 / length(v)); }

inline vec2 operator*(float a, const vec2& v) { return vec2(v.x * a, v.y * a); }

//--------------------------
struct vec3 {
	//--------------------------
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
	vec3(vec2 v) { x = v.x; y = v.y; z = 0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator-()  const { return vec3(-x, -y, -z); }
};

inline float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

inline float length(const vec3& v) { return sqrtf(dot(v, v)); }

inline vec3 normalize(const vec3& v) { return v * (1 / length(v)); }

inline vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline vec3 operator*(float a, const vec3& v) { return vec3(v.x * a, v.y * a, v.z * a); }

//--------------------------
struct vec4 {
	//--------------------------
	float x, y, z, w;

	vec4(float x0 = 0, float y0 = 0, float z0 = 0, float w0 = 0) { x = x0; y = y0; z = z0; w = w0; }
	vec4(vec3 xyz, float _w) { x = xyz.x;  y = xyz.y; z = xyz.z;  w = _w; }
	float& operator[](int j) { return *(&x + j); }
	float operator[](int j) const { return *(&x + j); }

	vec4 operator*(float a) const { return vec4(x * a, y * a, z * a, w * a); }
	vec4 operator/(float d) const { return vec4(x / d, y / d, z / d, w / d); }
	vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	vec4 operator-(const vec4& v)  const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
	vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
	void operator+=(const vec4 right) { x += right.x; y += right.y; z += right.z, w += right.z; }
};

inline float dot(const vec4& v1, const vec4& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
}

inline vec4 operator*(float a, const vec4& v) {
	return vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}

//---------------------------
struct mat4 { // row-major matrix 4x4
//---------------------------
	vec4 rows[4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		rows[0][0] = m00; rows[0][1] = m01; rows[0][2] = m02; rows[0][3] = m03;
		rows[1][0] = m10; rows[1][1] = m11; rows[1][2] = m12; rows[1][3] = m13;
		rows[2][0] = m20; rows[2][1] = m21; rows[2][2] = m22; rows[2][3] = m23;
		rows[3][0] = m30; rows[3][1] = m31; rows[3][2] = m32; rows[3][3] = m33;
	}
	mat4(vec4 it, vec4 jt, vec4 kt, vec4 ot) {
		rows[0] = it; rows[1] = jt; rows[2] = kt; rows[3] = ot;
	}

	vec4& operator[](int i) { return rows[i]; }
	vec4 operator[](int i) const { return rows[i]; }
	operator float* () const { return (float*)this; }
};

inline void matrixbolEulerFok3(int a, int b, int c,
	int d, int e, int f,
	int g, int h, int i) {
	vec3 fok = vec3(atan2f(h, i), atan2(-g, sqrtf(h * h + i * i)), atan2(d, a));
	printf("%3.05f %3.05f %3.05f \n", fok.x, fok.y, fok.z);

}

inline vec4 operator*(const vec4& v, const mat4& mat) {
	return v[0] * mat[0] + v[1] * mat[1] + v[2] * mat[2] + v[3] * mat[3];
}

inline mat4 operator*(const mat4& left, const mat4& right) {
	mat4 result;
	for (int i = 0; i < 4; i++) result.rows[i] = left.rows[i] * right;
	return result;
}

inline mat4 TranslateMatrix(vec3 t) {
	return mat4(vec4(1, 0, 0, 0),
		vec4(0, 1, 0, 0),
		vec4(0, 0, 1, 0),
		vec4(t.x, t.y, t.z, 1));
}

inline mat4 ScaleMatrix(vec3 s) {
	return mat4(vec4(s.x, 0, 0, 0),
		vec4(0, s.y, 0, 0),
		vec4(0, 0, s.z, 0),
		vec4(0, 0, 0, 1));
}


inline mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = normalize(w);
	return mat4(vec4(c * (1 - w.x * w.x) + w.x * w.x, w.x * w.y * (1 - c) + w.z * s, w.x * w.z * (1 - c) - w.y * s, 0),
		vec4(w.x * w.y * (1 - c) - w.z * s, c * (1 - w.y * w.y) + w.y * w.y, w.y * w.z * (1 - c) + w.x * s, 0),
		vec4(w.x * w.z * (1 - c) + w.y * s, w.y * w.z * (1 - c) - w.x * s, c * (1 - w.z * w.z) + w.z * w.z, 0),
		vec4(0, 0, 0, 1));
}


inline float degToRad(float deg) {
	return deg * M_PI / 180;
}



inline float pontesEgyenes(vec3 e1, vec3 e2, vec3 p) {
	float a, b, c;
	a = e2.y - e1.y;
	b = e1.x - e2.x;
	c = e2.x * e1.y - e1.x * e2.y;
	return (fabs(a * p.x + b * p.y + c) / sqrt(pow(a, 2) + pow(b, 2)));
}


struct Dnum {

	float value, derivative;

	Dnum(float f0 = 0, float d0 = 0) { value = f0, derivative = d0; }
	Dnum operator-() { return Dnum(-value, -derivative); }
	float& f() { return value; }
	float& d() { return derivative; }
};

inline Dnum operator+(Dnum l, Dnum r) { return Dnum(l.f() + r.f(), l.d() + r.d()); }
inline Dnum operator-(Dnum l, Dnum r) { return Dnum(l.f() - r.f(), l.d() - r.d()); }
inline Dnum operator*(Dnum l, Dnum r) { return Dnum(l.f() * r.f(), l.f() * r.d() + l.d() * r.f()); }
inline Dnum operator/(Dnum l, Dnum r) { return Dnum(l.f() / r.f(), (l.d() * r.f() - l.f() * r.d()) / r.f() / r.f()); }

// Elementary functions prepared for the chain rule as well
inline Dnum Sin(Dnum g) { return Dnum(sin(g.f()), cos(g.f()) * g.d()); }
inline Dnum Cos(Dnum g) { return Dnum(cos(g.f()), -sin(g.f()) * g.d()); }
inline Dnum Tan(Dnum g) { return Sin(g) / Cos(g); }
inline Dnum Log(Dnum g) { return Dnum(logf(g.f()), 1 / g.f() * g.d()); }
inline Dnum Exp(Dnum g) { return Dnum(expf(g.f()), expf(g.f()) * g.d()); }
inline Dnum Pow(Dnum g, float n) { return Dnum(powf(g.f(), n), n * powf(g.f(), n - 1) * g.d()); }


template<class T> struct DnumT {
	float f;
	T d;
	DnumT(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }

	DnumT operator+(DnumT r) { return DnumT(f + r.f, d + r.d); }
	DnumT operator*(DnumT r) { return DnumT(f * r.f, f * r.d + d * r.f); }
	DnumT operator/(DnumT r) { return DnumT(f / r.f, (d * r.f - f * r.d) / r.f / r.f); }

};

template<class T> DnumT<T> Sin(DnumT<T> g) { return DnumT<T>(sin(g.f()), cos(g.f()) * g.d()); }
template<class T> DnumT<T> Cos(DnumT<T> g) { return DnumT<T>(cos(g.f()), -sin(g.f()) * g.d()); }
template<class T> DnumT<T> Tan(DnumT<T> g) { return Sin(g) / Cos(g); }
template<class T> DnumT<T> Log(DnumT<T> g) { return DnumT<T>(logf(g.f()), 1 / g.f() * g.d()); }
template<class T> DnumT<T> Exp(DnumT<T> g) { return DnumT<T>(expf(g.f()), expf(g.f()) * g.d()); }
template<class T> DnumT<T> Pow(DnumT<T> g, float n) { return DnumT<T>(powf(g.f(), n), n * powf(g.f(), n - 1) * g.d()); }


//--------------------------
struct Complex {
	//--------------------------
	float x, y;

	Complex(float x0 = 0, float y0 = 0) { x = x0, y = y0; }
	Complex operator+(Complex r) { return Complex(x + r.x, y + r.y); }
	Complex operator-(Complex r) { return Complex(x - r.x, y - r.y); }
	Complex operator*(Complex r) { return Complex(x * r.x - y * r.y, x * r.y + y * r.x); }
	Complex operator/(Complex r) {
		float l = r.x * r.x + r.y * r.y;
		return (*this) * Complex(r.x / l, -r.y / l);
	}
};

Complex Polar(float r, float phi) {
	return Complex(r * cosf(phi), r * sinf(phi));
}


struct Quadrics {
	mat4 Q;
	float f(vec4 r) {
		return dot(r * Q, r);
	}
	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}
};


inline void evalHenger(float u, float v, vec3& point, vec3& normal, float height, float r) {
	float U = u * 2 * M_PI, V = v * height;
	printf("%3.05f\n", U);
	vec3 base(cos(U) * r, sin(U) * r, 0), spine(0, 0, V);
	point = base + spine;
	printf("%3.05f %3.05f %3.05f \n", point.x, point.y, point.z);
	normal = base;
	printf("%3.05f %3.05f %3.05f \n", normal.x, normal.y, normal.z);
}

inline void evalHiperboloid(float u, float v, vec3& point, vec3& normal, float h, float r) {
	float U = (v - 0.5f) * h, V = u * 2 * M_PI;
	float shu = sinh(U), chu = cosh(U), cv = cos(V), sv = sin(V);
	point = vec3(r * chu * cv, r * chu * sv, shu);
	printf("%3.05f %3.05f %3.05f \n", point.x, point.y, point.z);
	vec3 drdU(r * shu * cv, r * shu * sv, chu);
	vec3 drdV(r * chu * (-sv), r * chu * cv, 0);
	normal = cross(drdU, drdV);
	printf("%3.05f %3.05f %3.05f \n", normal.x, normal.y, normal.z);
}



//1. kviz//
inline void pontEgyenes(vec2 p, vec3 e) {
	printf("Tavolsag %3.05f\n", (abs(e.x * p.x + e.y * p.y + e.z)) / (sqrtf(e.x * e.x + e.y * e.y)));
	//return (abs(e.x * p.x + e.y * p.y + e.z)) / (sqrtf(e.x * e.x + e.y * e.y));
}

double toRad(double degree) {
	return degree / 180.0 * M_PI;
}

float toDeg(float rad) {
	return rad / M_PI * 180.0;
}

inline void varos(double lat1, double long1, double lat2, double long2, int radius) {
	double dist;
	dist = sin(toRad(lat1)) * sin(toRad(lat2)) + cos(toRad(lat1)) * cos(toRad(lat2)) * cos(toRad(long1 - long2));
	dist = acos(dist);
	//        dist = (6371 * pi * dist) / 180;
		//got dist in radian, no need to change back to degree and convert to rad again.
	dist = radius * dist;
	printf("Tavolsag %3.05f \n", dist);
	//return dist;
}

//Hiperbolikus sik tavolsag
inline void hyperbolicD(vec3 p, vec3 q) {
	printf("Tavolsag  %3.05f\n", acoshf(-1 * ((p.x * q.x) + (p.y * q.y) - (p.z * q.z))));
	//return acoshf(-1 * ((p.x * q.x) + (p.y * q.y) - (p.z * q.z)));
}

/*
Pythonban: https://www.programiz.com/python-programming/online-compiler/
////////////////////////////////////////////////////////////////////////////////////////////////////////////
Bezier:
p0 = (4, 5) # Változtasd meg a te értékeidre
p1 = (1, 2)
p2 = (6, 3)

t = 1 #ennek a kétszeresének kell lennie az értékednek

# Innentõl ne változtass, kattints a futtatás (jobbra mutató nyíl) gombra, alul kijön a válasz

x1 = p0[0] + t / 2 * (p1[0] - p0[0])
x2 = p1[0] + t / 2 * (p2[0] - p1[0])

print("Megoldás:", x1 + t / 2 * (x2 - x1))
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Catmull-Rom:
# Változtasd meg a saját értékeidre
p0 = (8, 9)
t0 = 0
p1 = (8, 6)
t1 = 1
p2 = (8, 4)
t2 = 2
p3 = (8, 9)
t3 = 3

t = 1.5

# Innentõl ne változtass, kattints a futtatás (jobbra mutató nyíl) gombra, alul kijön a válasz

class vec:
  def __init__(self, tup):
	self.x = tup[0]
	self.y = tup[1]

  def __add__(self, other):
	return vec((self.x + other.x, self.y + other.y))

  def __sub__(self, other):
	return vec((self.x - other.x, self.y - other.y))

  def __mul__(self, other):
	return vec((self.x * other, self.y * other))

  def __rmul__(self, other):
	return self * other

  def __truediv__(self, other):
	return vec((self.x / other, self.y / other))

  def __str__(self):
	return f"({self.x}, {self.y})"

cps = [vec(p) for p in [p0, p1, p2, p3]]
ts = [t0, t1, t2, t3]

def hermite(p0, v0, t0, p1, v1, t1, t):
  td = t1 - t0
  a0 = p0
  a1 = v0
  a2 = 3 * (p1 - p0) / (td ** 2) - (v1 + 2 * v0) / td
  a3 = 2 * (p0 - p1) / (td ** 3) + (v1 - v0) / (td ** 2)
  return a3 * (t - t0) ** 3 + a2 * (t - t0) ** 2 + a1 * (t - t0) + a0

def get_vel(i):
  return ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]) + (cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1])) / 2

for i in range(len(ts) - 1):
  if ts[i] <= t and t <= ts[i + 1]:
	v0 = get_vel(i)
	v1 = get_vel(i + 1)
	print("Megoldás:", hermite(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t).x)
////////////////////////////////////////////////////////////////////////////////////////////////////////////

Lagrange:
p0 = (4, 7)
t0 = 0
p1 = (5, 6)
t1 = 1
p2 = (7, 5)
t2 = 2

t = 1

# Innentõl ne változtass, kattints a futtatás (jobbra mutató nyíl) gombra, alul kijön a válasz

class vec:
  def __init__(self, tup):
	self.x = tup[0]
	self.y = tup[1]

  def __add__(self, other):
	return vec((self.x + other.x, self.y + other.y))

  def __sub__(self, other):
	return vec((self.x - other.x, self.y - other.y))

  def __mul__(self, other):
	return vec((self.x * other, self.y * other))

  def __rmul__(self, other):
	return self * other

  def __truediv__(self, other):
	return vec((self.x / other, self.y / other))

  def __str__(self):
	return f"({self.x}, {self.y})"

cps = [vec(p) for p in [p0, p1, p2]]
ts = [t0, t1, t2]

def L(i, t):
  li = 1
  for j in range(len(cps)):
	if i != j:
	  li *= (t - ts[j]) / (ts[i] - ts[j])
  return li

r = vec((0, 0))
for i in range(len(cps)):
  r = r + cps[i] * L(i, t)

print("Megoldás:", r.x)

*/



//2. kviz//
struct BezierCurve {
	std::vector<vec3> cps;

	float B(int i, float t) {
		int n = cps.size() - 1;
		float choose = 1;
		for (int j = 1; j <= i; j++)
			choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

	void AddControlPoint(vec3 cp) {
		cps.push_back(cp);
	}

	vec3 r(float t) {
		vec3 rt(0, 0, 0);
		for (int i = 0; i < cps.size(); i++)
			rt = rt + cps[i] * B(i, t);
		return rt;
	}
};


//Valamiért nem mindig megy
struct CatmullRom {
	std::vector<vec3> cps;
	std::vector<float> ts;
	vec3 Hermite(vec3 p0, vec3 v0, float t0,
		vec3 p1, vec3 v1, float t1, float t) {

		vec3 a0 = p0;
		vec3 a1 = v0;
		vec3 a2 = (3 * (p1 - p0)) / pow((t1 - t0), 2) - (v1 + 2 * v0) / (t1 - t0);
		vec3 a3 = (2 * (p0 - p1)) / pow((t1 - t0), 3) + (v1 + v0) / pow((t1 - t0), 2);
		vec3 r = a3 * pow((t - t0), 3) + a2 * pow((t - t0), 2) + a1 * pow((t - t0), 1) + a0;
		return r;
	}

	void AddControlPoint(vec3 cp, float t) {
		float ti = cps.size();
		cps.push_back(cp); ts.push_back(ti);
	}

	vec3 r(float t) {
		for (int i = 0; i < cps.size() - 1; i++)
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec3 v0 = 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]) + (cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1]));
				vec3 v1 = 0.5f * ((cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1]) + (cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]));
				return Hermite(cps[i], v0, ts[i],
					cps[i + 1], v1, ts[i + 1], t);
			}
	}
};


struct LagrangeCurve {
	std::vector<vec3>cps;
	std::vector<float>ts;

	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++)
			if (j != i)
				Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}

	void AddControlPoint(vec3 cp) {
		float ti = cps.size();
		cps.push_back(cp);
		ts.push_back(ti);
	}

	vec3 r(float t) {
		vec3 rt(0, 0, 0);
		for (int i = 0; i < cps.size(); i++)
			rt = rt + cps[i] * L(i, t);
		return rt;
	}
};



//3. kviz//
/*
* Ki kell szamolgatni az ABCDEF ertekeket a megadott pontok es kepuk alapjan.
*
* x’ = A*x + B*y + C
* y’ = D*x + E*y + F
*/
inline void affin(vec2 p, int A, int B, int C, int D, int E, int F) {
	printf("X: %3.05f Y: %3.05f \n", A * p.x + B * p.y + C, D * p.x + E * p.y + F);
}

//W (vagy S) az elsõként van megadva valszeg a feladatban
inline vec4 qmul(vec4 q1, vec4 q2) { //Kvaternió szorzás
	vec3 d1 = vec3(q1.x, q1.y, q1.z);
	vec3 d2 = vec3(q2.x, q2.y, q2.z);
	vec4 res = vec4(d2 * q1.w + d1 * q2.w + cross(d1, d2), q1.w * q2.w - dot(d1, d2));
	//printf("S: %3.05f X: %3.05f Y: %3.05f Z: %3.05f\n", res.w,res.x,res.y,res.z);
	return vec4(d2 * q1.w + d1 * q2.w + cross(d1, d2), q1.w * q2.w - dot(d1, d2));
}

/*
inline vec4 powVec(vec4 v, int pow) {
	return vec4(powf(v.x,pow), powf(v.y, pow), powf(v.z, pow), powf(v.w, pow));
}
*/

/*
Kvaternio invertalas matlabban:
quatinv([]);
pl.: qinv = quatinv([0 0 sqrt(2)/2 sqrt(2)/2]);
*/

inline vec4 quaternion(float ang, vec3 axis) { //Konstruálás
	vec3 d = normalize(axis) * sin(ang / 2);
	return vec4(d.x, d.y, d.z, cos(ang / 2));
}

inline vec3 Rotate(vec3 u, vec4 q) {
	vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
	return vec3(qr.x, qr.y, qr.z);
}



//Adott ket egyenes implicit egyenletukkel, szamutsuk ki a metszespont w harmadik homogen koordinatajat
inline void ketEgyenesImplicit(vec3 e1, vec3 e2) {
	printf("w: %3.05f\n", e1.x * e2.y - e1.y * e2.x);
}


//Egy sik implicit egyenlete 8x + 10y + 6y + 8 = 0 sik normalvektoraban az x y komponensek arany vagy 8/10
//Derivalni kell 1 fele keppen



//4. kviz//
inline void camera2d(vec2 kozeppont, float szelesseg, float magassag, vec2 pont) {
	vec2 result = vec2((pont.x - kozeppont.x) / magassag, (pont.y - kozeppont.y) / szelesseg);
	printf("%3.05f , %3.05f \n", result.x, result.y);
}


void DDA(int x1, int y1, int x2, int y2, int T_) {
	const int T = T_; // fractional bits
	int m = ((y2 - y1) << T) / (x2 - x1);
	int y = (y1 << T) + (1 << (T - 1)); // +0.5
	for (short x = x1; x <= x2; x++) {
		short Y = y >> T; // trunc
		y = y + m;
	}

	printf("%3.05f\n", m);
}


inline void atlokszama(int n) {
	printf("%3.05f\n", (n * (n - 3)) / 2);
}

//Egy szakasz ket vegpontja homogen koordinatakba 
//Mi lesz a szakasz es az Descartes koordokban x=1 egyenletu sik metszespontjanak y Descartes koordja?
//szakaszt meg sikot bebaszni geogebraba https://www.geogebra.org/3d
inline void szakaszketvegpont(vec4 p1, vec4 p2, int x) {
	vec3 ujp1, ujp2;
	ujp1 = vec3(p1.x / p1.w, p1.y / p1.w, p1.z / p1.w);
	ujp2 = vec3(p2.x / p2.w, p2.y / p2.w, p2.z / p2.w);
	vec3 res = ujp2 - ujp1;
	printf("X: %3.05f Y: %3.05f Z: %3.05f\n", res.x, res.y, res.z);
}


//5. kviz
//glDrawArrays(GL_TRIANGLE_FAN, 3,6) hany haromszoget rajzol ki? masodik attrib - 2


//6.kviz
/*
* haromszog csucsa modelezzesben es texturaban: hogyan fugg az xpix es ypix pixelkoord es texturakord, ha
* gViewport(0,0,1000,1000 et hivunk)
*
Textúra -> Model koordináta mátrixot lehet számolni
Textúra: [u, v]
Modell: [x,y,z]

[u, v, 1]    *M = [x,y,z]
[x,y,z,1] * T = [X,Y,Z,w]
[xpix, ypix] = [X / w, Y/w]
X = x /w Y = y/w

ViewPort transzformáció :
Ebben a formátumban várja az eredmény
xpix = wx * (xndc + 1) / 2 + cx
ypix = wy * (yndc + 1) / 2 + cy

*/
inline void haromszogMVP(vec3 m1, vec2 t1, vec3 m2, vec2 t2, vec3 m3, vec2 t3) {


	//elso oszlop
	//Három pont x koordinátája (0,0,1) és a két textúra koordináta (0,0), (0,1), (1,0) 
	//alapján az M mátrix elsõ oszlopa
	float elsoMCX = 0, elsoMBX = 0, elsoMAX = 0;
	elsoMCX = m1.x - elsoMAX * t1.x - elsoMBX * t1.y;
	elsoMBX = (m2.x - elsoMAX * t2.x - elsoMCX) / t2.y;
	elsoMAX = (m3.x - elsoMBX * t3.y - elsoMCX) / t3.x;


	//masodik oszlop
	//Három pont y koordinátája alapján az M mátrix elsõ oszlopa és a két textúra koordináta (0,0), (0,1), (1,0)
	//alapján az M mátrix második oszlopa
	float masodikMCY = 0, masodikMBY = 0, masodikMAY = 0;
	masodikMCY = m1.y - masodikMAY * t1.x - masodikMBY * t1.y;
	masodikMBY = (m2.y - masodikMAY * t2.x - masodikMCY) / t2.y;
	masodikMAY = (m3.y - masodikMBY * t3.y - masodikMCY) / t3.x;


	//harmadik oszlop
	//Három pont z koordinátája alapján az M mátrix elsõ oszlopa és a két textúra koordináta (0,0), (0,1), (1,0) 
	//alapján az M mátrix harmadik oszlopa
	float harmadikMCZ = 0, harmadikMBZ = 0, harmadikMAZ = 0;
	harmadikMCZ = m1.z - harmadikMAZ * t1.x - harmadikMBZ * t1.y;
	harmadikMBZ = (m2.z - harmadikMAZ * t2.x - harmadikMCZ) / t2.y;
	harmadikMAZ = (m3.z - harmadikMBZ * t3.y - harmadikMCZ) / t3.x;


	printf("M MATRIX\n %3.05f  %3.05f  %3.05f\n %3.05f  %3.05f  %3.05f\n %3.05f  %3.05f  %3.05f\nEZT MATRIX SZOROZNI [u,v,1] * M\n https://www.symbolab.com/\n",
		elsoMAX, masodikMAY, harmadikMAZ,
		elsoMBX, masodikMBY, harmadikMBZ,
		elsoMCX, masodikMCY, harmadikMCZ);

	//Ezutan ha megvan akkor a vegere rakni kell 1-et, hogy ne 1x3 legyne hanem 1x4 és szorozni jobbrol MVP-vel
	//pl ilyen eredmeny lesz: [u, v, 0.5u + 0.5v, -0.5u - 0.5v +1] = NDC
	//x = u / utolso koord
	//y = v / utolso koord
	//ismet:
	//xpix = wx * (xndc + 1) / 2 + cx
	//ypix = wy * (yndc + 1) / 2 + cy
	//xpix = 1000 * ( u / (-0.5u - 0.5v + 1) + 1) / 2 + cx
	//ypix = 1000 * ( v / (-0.5u - 0.5v + 1) + 1) / 2 + cy
	//Ezt a fenti 2-t kell egyszeruseteni
}	//https://www.symbolab.com/solver/step-by-step/%5Cbegin%7Bpmatrix%7Du%26v%260.5u%2B0.5v-1%261%5Cend%7Bpmatrix%7D%5Ccdot%5Cbegin%7Bpmatrix%7D1%260%260%260%5C%5C%20%20%200%261%260%260%5C%5C%20%20%200%260%261%26-1%5C%5C%20%20%200%260%261%260%5Cend%7Bpmatrix%7D%20

//7. kviz//
inline float F(float n, float k) {
	printf("Szazalekban: %3.05f", ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k) * 100);
	return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
}

inline vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (vec3(1.0, 1.0, 1.0) - F0) * pow(cosTheta, 5);
}

//N = normalvektor L = irany Lin = 9 W/m^2/st sugarsuruseeg kd = diffuz visszaverodesi tenyezo
//Itt a detekalt pontot nemvagom (1,2,3) nal jol mukodott
inline float sugarsurusegDiffuz(vec3 N, vec3 L, float Lin, float kd) {
	vec3 normalizedL = normalize(L);
	float costheta = dot(N, normalizedL);
	printf("%3.05f \n", Lin * kd * costheta);
	return Lin * kd * costheta;
}

inline vec3 abs(vec3 vec) {
	return vec3(fabs(vec.x), fabs(vec.y), fabs(vec.z));
}

//N = normalvektor L = irany Lin = 9 W/m^2/st sugarsuruseeg ks = spekularis visszaverodesi tenyezo , shine shininess
inline float sugarsurusegSpekularis(vec3 N, vec3 L, float Lin, float ks, vec3 V, float shine) {
	vec3 H = (L + V) / (length(L + V));
	float cosdelta = dot(N, H);
	printf("%3.05f \n", V.y * ks * powf(cosdelta, shine));
	return V.y * ks * powf(cosdelta, shine);
}

//Fenysugar egy 1.0 toresmutatoju kozegbol erkezika  kozeg hatarara, hany fokos szoget kell bezarnia 
//a fenysugar iranyanak es a levego fele mutato feluletu normalisnak, 
//hogy a fenysugarbol semmi se tudjon kilepni a kozegbol es teljes visszaverodes legyen
//Ha 1.0/0.3 alaku akkor siman 0.3
inline float fenysugarKozegbol(float n) {
	printf("Fok: %3.05f\n", toDeg(asinf(n)));
	return toDeg(asinf(n));
}


//9. kviz

//hany csucspontbol fog allni a VBO
inline void gltriangleStrip(int n) {
	printf("%d\n", (n - 1) * (n * 2));
}

inline void gltriangles(int n) {
	printf("%d\n", 2 * ((n - 2) * 3 + 3) + (n - 2) * ((n - 2) * 6 + 6));
}


//Normalvektor komponensek
inline void parameteresFelulet(float a, float b, float c, float d, float e, float f, float g, float h, float i, float u, float v) {
	float nx = (d + c * v) * (h + c * u) - (g + c * v) * (e + c * u);
	float ny = (g + i * v) * (b + c * u) - (a + c * v) * (h + i * u);
	float nz = (a + c * v) * (e + f * u) - (d + f * v) * (b + c * u);
	printf("nx: %3.05f ny: %3.05f nz: %3.05f \n", nx, ny, nz);
}

//Haromszog harom csucsa kepernyo, mennyivel valtozik a Z koordinata ha jobboldali szomszedra lepunk?
inline void haromszogCsucs(vec3 cs1, vec3 cs2, vec3 cs3) {
	vec3 n = cross(cs3 - cs1, cs2 - cs1);
	printf("nx: %3.05f ny: %3.05f nz: %3.05f \n nx/nz %3.20f\n", n.x, n.y, n.z, fabs(n.x / n.z));
}


//10. kviz

//Legalabb mennyi egesz azt kell nezni
inline void parsecPersec(vec3 p1, vec3 s1, vec3 p2, vec3 s2, float dt) {
	vec3 newP1 = p1 + dt / 1000 * s1;
	vec3 newP2 = p2 + dt / 1000 * s2;
	vec3 diff = newP2 - newP1;
	float distance = sqrtf(dot(diff, diff));
	printf("%3.05f\n", distance);
}


inline void testElek(int darab, int csucs, int lap) {
	printf("%d\n", csucs + lap - darab * 2); //ennyi ele van "darab" poliederre
	//1 poliederre csucs + lap = el + 2
}


//Terep irányvektora egy x = 5 és y = 7 szintvonal érintõ !irányvektorát!, ha normál akkor elég a gradiens magában
//ttps://www.wolframalpha.com/input/?i=x%5E2+%2B+3+y%5
//ezt megfordítod, szorzod -1 el az egyik koordot és megvan



/*
Billboard matlabban:

scale = [1,1,1];
eye = [4,2,7];
pos = [1,2,3];
up = [0,1,0];

w = eye - pos;
r = cross(w,up);
u = cross(r,w);
r = r/norm(r).*scale;
u = u/norm(u).*scale;

rslt = [r(1), r(2), r(3), 0;
		u(1), u(2), u(3), 0;
		0,    0,    1,    0;
		pos(1), pos(2), pos(3), 1]


*/

//FPS jatek, FRENET
inline void FPS(vec3 pos, vec3 seb, vec3 gyors) {
	vec3 w, v, u;
	w = seb * (-1.0);
	v = gyors;
	u = cross(v, w);
	w = normalize(w);
	v = normalize(v);
	u = normalize(u);

	printf("%3.05f %3.05f %3.05f 0\n", u.x, v.x, w.x);
	printf("%3.05f %3.05f %3.05f 0\n", u.y, v.y, w.y);
	printf("%3.05f %3.05f %3.05f 0\n", u.z, v.z, w.z);
	printf("%3.05f %3.05f %3.05f 1\n", pos.x, pos.y, pos.z);
}

//xy síkra billboard referencia
inline void billBoard(vec3 eye, vec3 pos, vec3 up) {
	vec3 w = eye - pos;
	vec3 r = normalize(cross(up, w));
	printf("%3.05f %3.05f %3.05f 0\n", r.x, r.y, r.z);

	vec3 u = normalize(cross(w, r));
	printf("%3.05f %3.05f %3.05f 0\n", u.x, u.y, u.z);

	printf("0 0 1 0\n");
	printf("%3.05f %3.05f %3.05f 0\n", pos.x, pos.y, pos.z);
}