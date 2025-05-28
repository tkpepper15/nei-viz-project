/**
 * Complex number class for impedance calculations
 */
export class Complex {
    constructor(public real: number, public imaginary: number) {}

    static add(a: Complex, b: Complex): Complex {
        return new Complex(a.real + b.real, a.imaginary + b.imaginary);
    }

    static subtract(a: Complex, b: Complex): Complex {
        return new Complex(a.real - b.real, a.imaginary - b.imaginary);
    }

    static multiply(a: Complex, b: Complex): Complex {
        return new Complex(
            a.real * b.real - a.imaginary * b.imaginary,
            a.real * b.imaginary + a.imaginary * b.real
        );
    }

    static divide(a: Complex, b: Complex): Complex {
        const denom = b.real * b.real + b.imaginary * b.imaginary;
        return new Complex(
            (a.real * b.real + a.imaginary * b.imaginary) / denom,
            (a.imaginary * b.real - a.real * b.imaginary) / denom
        );
    }

    static reciprocal(a: Complex): Complex {
        const denom = a.real * a.real + a.imaginary * a.imaginary;
        return new Complex(a.real / denom, -a.imaginary / denom);
    }

    static magnitude(a: Complex): number {
        return Math.sqrt(a.real * a.real + a.imaginary * a.imaginary);
    }

    static phase(a: Complex): number {
        return Math.atan2(a.imaginary, a.real);
    }
}

/**
 * Helper to create complex number from real and imaginary parts
 */
export const createComplex = (real: number, imaginary: number): Complex => 
    new Complex(real, imaginary);

/**
 * Create complex number from magnitude and phase
 */
export const fromPolar = (magnitude: number, phase: number): Complex => 
    new Complex(
        magnitude * Math.cos(phase),
        magnitude * Math.sin(phase)
    ); 