const nodes = Array.from(document.getElementsByTagName("h1"));
const cache = {
	viewport: {},
	rects: []
};

// init
window.addEventListener("load", init);

function init() {
	// update the cache and check scroll position
	recache();

	// throttle the scroll callback for performance
	document.addEventListener("scroll", throttle(scrollCheck, 10));

	// debounce the resize callback for performance
	window.addEventListener("resize", debounce(recache, 50));
};

// update the cache and check scroll position
function recache() {
	// cache the viewport dimensions
	cache.viewport = {
			width: window.innerWidth,
			height: window.innerHeight
	};

	// cache node dimensions because we don't really want to
	// call getBoundingClientRect() during scroll - even when throttled
	nodes.forEach((node, i) => {
		cache.rects[i] = rect(node);
	});

	scrollCheck();
}

// check whether a node is at or above the horizontal halfway mark
function scrollCheck() {
	// instead of relying on calling getBoundingClientRect() everytime,
	// let's just take the cached value and subtract the pageYOffset value
	// and see if the result is at or above the horizontal midline of the window
	const offset = getScrollOffset();
	const midline = cache.viewport.height * 0.5;

	cache.rects.forEach((rect, i) => {
		nodes[i].classList.toggle("active", rect.y - offset.y < midline);
	});
};

// get the scroll offsets
function getScrollOffset() {
	return {
		x: window.pageXOffset,
		y: window.pageYOffset
	};
};

// throttler
function throttle(fn, limit, context) {
	let wait;
	return function() {
		context = context || this;
		if (!wait) {
			fn.apply(context, arguments);
			wait = true;
			return setTimeout(function() {
				wait = false;
			}, limit);
		}
	};
};

// debouncer
function debounce(fn, limit, u) {
	let e;
	return function() {
		const i = this;
		const o = arguments;
		const a = u && !e;
		clearTimeout(e),
			(e = setTimeout(function() {
			(e = null), u || fn.apply(i, o);
		}, limit)),
			a && fn.apply(i, o);
	};

}

// getBoundingClientRect with offsets
function rect(e) {
	const o = getScrollOffset();
	const r = e.getBoundingClientRect();

	return {
			x: r.left + o.x,
			y: r.top + o.y
	};
};
