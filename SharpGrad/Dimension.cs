using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad
{
    public class Dimension(int size)
    {
        public int Size { get; } = size;
    }
}