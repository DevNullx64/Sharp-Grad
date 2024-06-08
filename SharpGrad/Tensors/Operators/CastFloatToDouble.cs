namespace SharpGrad.Tensors.Operators
{
    internal class CastFloatToDouble : IExecutor1<float, double>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;
        public static double Exec(float operand1) => operand1;
        public static float Backward(float operand1, double grad) => (float)grad;
    }

}