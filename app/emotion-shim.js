// This is a shim to provide basic @emotion/styled functionality
// when the actual package can't be loaded due to dependency conflicts

const styled = (Component) => {
  return (props) => {
    return Component(props);
  };
};

// Export a mock version of styled
export default styled;

// Mock serialize function
export const serializeStyles = () => ({
  name: 'mock-style',
  styles: '',
  map: null,
}); 